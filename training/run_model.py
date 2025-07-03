
import os
import sys

def run():
    from BorhNet.token2fr.model import Model
    from environment.verifier import Verifier
    
    FEYNRULES_PATH = "/users/qniu3/physics/FeynRules"
    FEYNARTS_PATH = "/users/qniu3/physics/FeynArts-3.12/Models" 
    NLOCT_PATH = "/users/qniu3/physics/FeynRules"
    MADGRAPH5_PATH = "/users/qniu3/physics/MG5_aMC_v3_6_3"

    JSON_PATH = sys.path[0] + '/SM.json'
    OUTPUT_PATH = '/oscar/home/qniu3/physics/RL_model_builder/theories'
 
    model = Model("Standard Model", "Cooper Niu", JSON_PATH, OUTPUT_PATH)
    print(model.score)
    model.to_fr()

    # verifier = Verifier(
    #     feynrules_path = FEYNRULES_PATH,
    #     feynarts_path = FEYNARTS_PATH,
    #     nloct_path = NLOCT_PATH,
    #     madgraph5_path = MADGRAPH5_PATH,
    #     model_path = model.output_dir,
    #     model_name = model.model_symbol
    #     )

    # verifier.output_UFO()


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run()
