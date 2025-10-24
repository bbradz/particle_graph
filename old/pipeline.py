import os
import csv
import time
import torch
import subprocess
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

"""
Design Dimensions to Explore--

1. Just output the text of a FeynRule file...
    Seems a bit unlikely it would do well
2. Do we need Verifier at test-time?
    **Integrating constants into Bohr output** 
        Verifier is only used in training to assess how good Bohr's coefficients were)
        Some ppl say the constants should be in the process (https://arxiv.org/pdf/2204.10532)
        Main algo "SymFormer" does seperate it out though (https://erik-derner.github.io/research/files/vastl2024symformer.pdf)
3. Train NN over multiple graph to reduce overfitting...

Bohr -> outputs particle model to
Einstein -> outputs tests to apply particle model to
Verifier -> outputs tuned coefficients for particle model
Bohr <- selfplay -> Einstein

Features:
- Self-Play between Bohr & Einstein
- Swappable Bohr architectures
- Swappable Verifer architectures
- Can turn off Einstein
- Customize ratio of epoch to train each model
- Able to swap LR, model dims, and optimizer in model class
- Save info to CSV
- Logs to Tensorboard

To Fill In:
- Can get estimate of uncertainty through dropout and mean_pred over multiple 
  preds and use std as measure of uncertainty
"""

class BaseModel(nn.Module):
    def __init__(self, name, optimizer_class=torch.optim.SGD, optimizer_params={'lr': 0.01}):
        super().__init__()
        self.name = name
        self.optimizer = optimizer_class(self.parameters(), **optimizer_params)
        self.buffer = {}
    def forward(self, x):
        raise NotImplementedError
    def mc_dropout_pred(self, s, n):
        raise NotImplementedError
    def loss(self, *args):
        raise NotImplementedError
    def finetune(self, dataset):
        self.optimizer.zero_grad()
        for data in dataset:
            loss = self.loss(*data)
            loss.backward()
        self.optimizer.step()
    def update(self, data, num_epochs=1):
        self.log_data({
            'pre-weights': self.get_weights(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'n_updates': num_epochs
        })
        for _ in range(num_epochs):
            loss = self.loss(*data)
            self.log_data({'loss': loss.item()})
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.log_data({'post-weights': self.get_weights()})
    def get_weights(self):
        return [param.data.clone().detach().flatten() for param in self.parameters()]
    def log_data(self, data_dict):
        for key, value in data_dict.items():
            if value is not None:
                self.buffer.setdefault(key, []).append(value)
    def clear_buffer(self):
        self.buffer = {}

class BohrModel(BaseModel):
    def __init__(self):
        super().__init__("Bohr")
    def forward(self, x):
        pass
    def loss(self, br):
        return torch.tensor(-br, dtype=torch.float, requires_grad=True)

class EinsteinModel(BaseModel):
    def __init__(self, test_count):
        super().__init__("Einstein")
        self.test_count = test_count
    def forward(self, s):
        pass
    def loss(self, p_m, target_p_m):
        return nn.MSELoss()(p_m, target_p_m)

class VerifierModel(BaseModel):
    def __init__(self):
        super().__init__("Verifier")
    def forward(self, s):
        pass
    def loss(self, v):
        return torch.tensor(-v, dtype=torch.float, requires_grad=True)

def compute_metrics(buffer):
    metrics = {}
    if not buffer:
        return metrics
    def safe_mean(values):
        return np.mean(values) if values else None
    for key, value in buffer.items():
        if not value:
            continue
        if key == 'sequence':
            metrics['len(sequence)'] = [len(seq) for seq in value]
            metrics['avg_len_sequence'] = safe_mean(metrics['len(sequence)'])
        elif key == 'logits' and 'post-weights' in buffer and 'pre-weights' in buffer:
            metrics['entropy of logits'] = -(torch.tensor(value) * torch.log(torch.tensor(value) + 1e-9)).sum(dim=-1).mean().item()
            metrics['l1_norm of weights'] = torch.norm(torch.cat(buffer['post-weights']), p=1).item()
            metrics['l2_norm of weights'] = torch.norm(torch.cat(buffer['post-weights']), p=2).item()
        elif key == 'loss':
            metrics['avg_loss'] = safe_mean(value)
        elif key == 'reward':
            metrics['avg_reward'] = safe_mean(value)
            metrics['reward_diff'] = safe_mean(np.diff(value)) if len(value) > 1 else 0.0
        elif key in ('inference start time', 'backward start time') and key.replace('start', 'end') in buffer:
            time_key = key.replace(' start time', '')
            metrics[f'time_for_{time_key}'] = safe_mean(np.array(buffer[key.replace('start', 'end')]) - np.array(buffer[key]))
        elif key == 'target_logits' and 'logits' in buffer:
            metrics['entropy of target_logits'] = -(torch.tensor(value) * torch.log(torch.tensor(value) + 1e-9)).sum(dim=-1).mean().item()
        elif key in ('clip_fraction', 'clip_range', 'approx_kl from backward'):
            metrics[key] = safe_mean(value)
    return metrics

def save_to_csv(filename, data, headers):
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        if file.tell() == 0:
            writer.writeheader()
        flat_data = [{k: ','.join(map(str, v)) if isinstance(v, list) else v for k, v in row.items()} for row in data]
        writer.writerows(flat_data)

def save_csvs(models, tensorboard=False, writer=None):
    if tensorboard and writer is None:
        writer = SummaryWriter()
    for model in models:
        if not model:
            continue
        metrics = compute_metrics(model.buffer)
        save_to_csv(f'{model.name}_buffer.csv', [{k: v for k, v in model.buffer.items()}], model.buffer.keys())
        if tensorboard:
            for key, value in metrics.items():
                if value is not None:
                    writer.add_scalar(f'{model.name}/{key}', value)
        model.clear_buffer()

def train_bohr(bohr, einstein, verifier, eps, all_tests, cutoff, eos, num_ver_guesses, bwd_epochs, tensorboard):
    writer = SummaryWriter() if tensorboard else None
    for t in range(1, eps + 1):
        s = []
        inference_start_time = time.time()
        while not s or s[-1] != eos:
            s = bohr(s)
        inference_end_time = time.time()
        bohr.log_data({'sequence': s, 'episode': t,
                       'inference start time': inference_start_time,
                       'inference end time': inference_end_time})
        
        rejected, pass_ratio = evaluate_rejection(s, einstein, all_tests, eos)
        br = (compute_verifier_reward(s, verifier, cutoff, num_ver_guesses, bwd_epochs, t)
              if not rejected else pass_ratio)

        backward_start_time = time.time()
        bohr.update([br], num_epochs=bwd_epochs[0](t))
        backward_end_time = time.time()
        bohr.log_data({'reward': br,
                       'backward start time': backward_start_time,
                       'backward end time': backward_end_time})

        if einstein:
            target_p_m = generate_target_probabilities(s, all_tests)

            einstein_inference_start = time.time()
            einstein_output = einstein(s)
            einstein_inference_end = time.time()
            einstein.log_data({'sequence': s, 'episode': t,
                               'target_logits': target_p_m.tolist(),
                               'inference start time': einstein_inference_start,
                               'inference end time': einstein_inference_end})

            einstein_backward_start = time.time()
            einstein.update([einstein_output, target_p_m], num_epochs=bwd_epochs[1](t))
            einstein_backward_end = time.time()
            einstein.log_data({'backward start time': einstein_backward_start,
                               'backward end time': einstein_backward_end})

            save_csvs([einstein], tensorboard=tensorboard, writer=writer)
        save_csvs([bohr, verifier], tensorboard=tensorboard, writer=writer)

    if writer:
        writer.close()

def generate_target_probabilities(s, all_tests):
    target_p_m = [1.0 if can_reject(s, test) else 0.0 for test in all_tests[:-1]]
    target_p_m.append(0.0 if any(target_p_m) else 1.0)
    return nn.Softmax(dim=0)(torch.tensor(target_p_m, dtype=torch.float))

def evaluate_rejection(s, einstein, all_tests, eos):
    if einstein:
        einstein_output = einstein(s)
        einstein.log_data({'logits': einstein_output.tolist()})
        
        sorted_tests_probs = get_sorted(einstein_output)
        cumm_prob, cumm_seen, cumm_passed, rejected = 0.0, 0, 0, False
        for test_idx, prob in sorted_tests_probs:
            test = all_tests[test_idx]
            if test == all_tests[-1]:
                break
            if can_reject(s, test):
                rejected = True
            else:
                cumm_passed += 1
            cumm_seen += 1
            cumm_prob += prob
            if rejected or cumm_prob >= 0.5:
                break
        return rejected, (cumm_passed / cumm_seen) if cumm_seen > 0 else 0.0
    else:
        results = [1 if can_reject(s, test) else 0 for test in all_tests[:-1]]
        return any(results), (sum(results) / (len(results) + 1)) if results else 0.0

def compute_verifier_reward(s, verifier, cutoff, num_ver_guesses, bwd_epochs, t):
    best_validity = float('-inf')
    for _ in range(num_ver_guesses):
        inference_start_time = time.time()
        s_c_prime = verifier(s)
        inference_end_time = time.time()
        verifier.log_data({'sequence': s, 'coefficients': s_c_prime,
                           'inference start time': inference_start_time,
                           'inference end time': inference_end_time})

        theory_validity = check_coeff_score(s_c_prime)
        vr = 1 + (theory_validity - cutoff) if theory_validity >= cutoff else (-1 if grid_search(s) else 0)

        backward_start_time = time.time()
        verifier.update([vr], num_epochs=bwd_epochs[2](t))
        backward_end_time = time.time()
        verifier.log_data({'reward': vr,
                           'backward start time': backward_start_time,
                           'backward end time': backward_end_time})
        best_validity = max(best_validity, theory_validity)

    final_reward = 1 + (best_validity - cutoff) if best_validity >= cutoff else (1 if grid_search(s) else 0.5)
    return final_reward

def convert_seq_to_fr(seq):
    pass

def convert_fr_to_ufo(fr_path):
    script = f'''
    $FeynRulesPath = SetDirectory["{feynrules_directory_path}"];
    LoadModel["{fr_path}"];
    WriteUFO[LSM, Output -> "SM_NLO"]
    Exit[]
    '''
    process = subprocess.run(["math", "-script", "-noinit"],
                            input=script, text=True, capture_output=True)
    os.makedirs(ufo_output_dir, exist_ok=True)
    return os.path.exists(ufo_output_dir)

def run_madgraph(ufo_path):
    script = f'''
    launch {ufo_path}
    set auto_convert_model T
    generate p p > t t~ [QCD]
    output mg5_output
    launch mg5_output
    set nevents = 100
    exit
    '''
    subprocess.run(["mg5_aMC"], input=script, text=True, capture_output=True)
    pass

def change_coeff(seq):
    pass

def get_sorted(p_m):
    sorted_indices = torch.argsort(p_m, descending=True)
    return [(index.item(), p_m[index].item()) for index in sorted_indices]

def can_reject(s, test):
    fr_path = convert_seq_to_fr(s)
    script = f'''
    SetDirectory["{feynrules_directory_path}"];
    LoadModel["{fr_path}"];
    {test};
    Exit[]
    '''
    subprocess.run(["math", "-script", "-noinit"],
                            input=script, text=True, capture_output=True)
    pass 

def grid_search(s, tests=None):
    iterations = 10
    for _ in range(iterations):
        s_new = change_coeff(s)
        r_s_new = run_madgraph(convert_fr_to_ufo(convert_seq_to_fr(s_new)))
        if r_s_new:
            return True
    return False

def check_coeff_score(s_c_prime):
    pass

if __name__ == '__main__':
    feynrules_directory_path = "/Users/benbradley/Downloads/feynrules"
    ufo_output_dir = "./ufo_output/SM_NLO"  
    custom_tokens = [
        'NODE', 'EDGE', 'HYPERNODE', 'ID', '-', 'mu', 'tau', 've', 'vm', 'vt',
        'u', 'd', 'c', 's', 't', 'b', 'g', 'a', 'Z', 'W+', 'W-', 'H'
    ]
    eos = "EOS"
    custom_tokens.append(eos)
    all_tests = [
        "CheckHermiticity[LSM]",
        "CheckHermiticity[LSM, FlavorExpand -> True]",
        "CheckMassSpectrum[LSM]",
        "CheckKineticTermNormalisation[LSM, FlavorExpand -> SU2W]",
        "CheckKineticTermNormalisation[LSM, FlavorExpand -> True]",
        "none"
    ]
    bohr_model = BohrModel()
    einstein_model = EinsteinModel(test_count=len(all_tests))
    verifier_model = VerifierModel()
    eps = 2
    cutoff = 0.8
    num_ver_guesses = 1
    bwd_epochs = [
        lambda t: 1,
        lambda t: 1,
        lambda t: 1
    ]
    tensorboard = False

    train_bohr(bohr_model, einstein_model, verifier_model, eps, all_tests,
               cutoff, eos, num_ver_guesses, bwd_epochs, tensorboard)