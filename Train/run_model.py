
import os
import sys

def run():
    from BorhNet.token2fr.model import Model

    JSON_PATH = sys.path[0] + '/SM.json'
    MODEL_PATH = ''

    model = Model("Standard Model", "Cooper Niu", JSON_PATH)
    print(model.score)
    #model.to_fr("TheoryArchive")


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(sys.path[0])

    run()
