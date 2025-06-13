
import os
import sys

def run():
    from BorhNet.token2fr.model import Model

    JSON_PATH = sys.path[0] + '/SM.json'
    OUTPUT_PATH = '/oscar/home/qniu3/physics/RL_model_builder/theories'
 
    model = Model("Standard Model", "Cooper Niu", JSON_PATH, OUTPUT_PATH)
    model.to_fr()
    print(model.score)

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run()
