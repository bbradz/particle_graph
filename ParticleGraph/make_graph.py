import json 

def make_graph(model_file):
    with open(model_file, 'r') as f:
        model = json.load(f)

    
    return model


if __name__ == "__main__":
    model_file = "./SM_test.json"
    model = make_graph(model_file)
    print(model)