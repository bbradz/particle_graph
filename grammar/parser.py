from grammar.vocab import GRAMMAR_TOKENS
from grammar.vocab import encode
import grammar.vocab as vocab

def read_model_txt(file):
    seq = []
    with open(file, "r") as f:
        sm_txt = f.read()
        for line in sm_txt.split("\n"):
            if not line.startswith("#"):
                line = line.split("#")[0].strip()
                if not line:
                    continue
                seq.extend(line.split())
    # Only add BOS/EOS if they don't already exist
    if not seq or seq[0] != "BOS":
        seq.insert(0, "BOS")
    if not seq or seq[-1] != "EOS":
        seq.append("EOS")
    return seq

def print_model(model):
    for key, value in model.items():
        print(f"\n ==== {key} ====")
        for k, v in value.items():
            print(f"  {v}")

def print_particles(particles):
    print("="*20)
    for key, value in particles.items():
        for color, charge in value.items():
            for charge, value in charge.items():
                print(key, color, charge, value)

def get_int(value):
    return int(value.split("_")[1])

def get_float(value):
    return float(value.split("_")[1])