import random
from grammar.masker import GrammarMasker
from config import config
import grammar.vocab as vocab
from grammar.parser import read_model_txt, print_model

import grammar.syntax as syntax

def generate_data(grammar_masker: GrammarMasker, 
                  seed: int = None,
                  prompt: bool = True,
                  max_length: int = config.max_length,
                  ) -> list[str]:

    state = grammar_masker.init_state()
    if seed is not None: random.seed(seed)
    sequence = ["BOS"]
    prev_token = "BOS"
    probability = 1

    if prompt: 
        model = read_model_txt("dataset/topless.txt")

        tokens = model
        if tokens and tokens[0] == "BOS":
            tokens = tokens[1:]
        if tokens and tokens[-1] == "EOS":
            tokens = tokens[:-1]

        # Step through the incomplete sequence to build up the grammar state
        for token in tokens:
            valid_tokens = grammar_masker.get_valid_tokens(state, prev_token)

            if not valid_tokens:
                print(f"Warning: No valid tokens after {prev_token}")
                break
            if token not in valid_tokens: 
                print(f"Warning: Token '{token}' is not valid after {prev_token}")
                break
            sequence.append(token)
            state = grammar_masker.step(state, token)
            prev_token = token

    while state.length < max_length:
        if prev_token == "EOS": break

        valid_tokens = grammar_masker.get_valid_tokens(state, prev_token)
        if not valid_tokens:
            print(f"Warning: No valid tokens after {prev_token}")
            break
        if valid_tokens[0] in vocab.GENS:
            # pick the preferred gen based on the number of particles
            color = "NO_COLOR" if state.rep_list[0] == "singlet" else "COLOR"
            gen_tokens = syntax.get_preferred_gen(state.chirality, color, state.charge_list, state.particles)
            token = random.choice(gen_tokens)
        else:
            token = random.choice(valid_tokens)
        sequence.append(token)
        state = grammar_masker.step(state, token)
        prev_token = token
        print(valid_tokens)
        print(token) 
        probability *= len(valid_tokens)

        full_model = {"gauge_groups": state.gauge_groups, 
                      "particles": state.particles,
                      "multiplets": state.multiplets, 
                      "interactions": state.interactions}
    print(f"Probability: {probability:.1e}")
    return sequence, full_model

if __name__ == "__main__":
    grammar_masker = GrammarMasker()
    sequence, full_model = generate_data(grammar_masker, prompt=True)
    print_model(full_model)

    