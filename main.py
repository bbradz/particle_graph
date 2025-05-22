if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from json2fr.model import Model

    JSON_PATH = 'theory_data/SM.json'
    MODEL_PATH = 'theory_data'

    model = Model("Standard Model", "Cooper Niu", JSON_PATH)
    model.to_fr(MODEL_PATH)


