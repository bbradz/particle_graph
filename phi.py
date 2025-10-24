import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from bitsandbytes.optim import AdamW

# --- CORRECTED MODEL NAME ---
MODEL_NAME = "microsoft/Phi-4-mini-reasoning"

def load_model_and_tokenizer(model_name: str):
    """
    Loads the specified Hugging Face model and tokenizer with 4-bit quantization.
    """
    print(f"Loading model: {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration for 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, # Use bfloat16 for mixed-precision
        device_map=device,
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer, device

def run_inference(model, tokenizer, device, prompt_text: str):
    """
    Runs inference using the model's proper chat template.
    """
    print("\n--- Running Inference ---")
    print(f"User Prompt: '{prompt_text}'")
    
    model.eval()

    # Apply the chat template - this is the correct way to format input
    messages = [{"role": "user", "content": prompt_text}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_new_tokens=30, # Increased for a more complete answer
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the newly generated tokens
    response_ids = outputs[0][inputs.shape[-1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    print(f"Model Output: '{response}'")
    print("------------------------")
    # Clear CUDA cache after inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def train_step_with_reward(model, tokenizer, device, optimizer, user_prompt: str, desired_completion: str, reward: float):
    """
    Performs one step of RL-style training using the chat template.
    """
    model.train()
    
    # 1. Prepare the full sequence using the chat template
    messages = [{"role": "user", "content": user_prompt}]
    
    # The full input includes the prompt and desired answer
    full_input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(device)

    # The prompt part ends just before the model's turn to speak
    prompt_token_len = full_input_ids.shape[1]

    # Now, add the desired completion
    completion_ids = tokenizer(desired_completion, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    
    # Combine them to form the full training sequence
    input_ids = torch.cat([full_input_ids, completion_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    
    # 2. Get model's predictions (logits)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # 3. Calculate the log probability of the desired completion
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    true_token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)

    # 4. Mask the loss to only apply to the completion tokens
    loss_mask = torch.zeros_like(shift_labels, dtype=torch.float)
    loss_mask[:, (prompt_token_len - 1):] = 1.0 # -1 because of the shift
    
    # 5. Define the RL-style loss
    masked_log_probs = true_token_log_probs * loss_mask
    log_prob_of_completion = masked_log_probs.sum()
    
    loss = -log_prob_of_completion * reward
    
    # 6. Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# --- Main Execution ---
if __name__ == "__main__":
    model, tokenizer, device = load_model_and_tokenizer(MODEL_NAME)

    inference_prompt = "The primary purpose of a particle accelerator is to"
    run_inference(model, tokenizer, device, inference_prompt)

    print("\n--- Starting Training ---")
    
    train_prompt = "The primary purpose of a particle accelerator is to"
    desired_completion = "accelerate charged particles, such as electrons or protons, to very high speeds and energies."
    positive_reward = 1.0
    
    # Use a memory-efficient optimizer from bitsandbytes
    optimizer = AdamW(model.parameters(), lr=5e-6, is_paged=True, optim_bits=8)
    
    num_training_steps = 1_000
    for i in range(num_training_steps):
        loss = train_step_with_reward(
            model, tokenizer, device, optimizer, 
            train_prompt, desired_completion, positive_reward
        )
        print(f"Step {i+1}/{num_training_steps}, Loss: {loss:.4f}")
        # Clear CUDA cache periodically during training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("------------------------")

    run_inference(model, tokenizer, device, inference_prompt)