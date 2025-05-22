import json

if __name__ == "__main__":
    FILE_PATH = "./theory_data/SM.json"
    with open(FILE_PATH, "r") as f:
        data = json.load(f)

    for i in data['GaugeGroups']:
        print("--------------------------------")
        for key, value in i.items():
            print(key, value)

    for i in data['particles']:
        print("--------------------------------")
        for key, value in i.items():
            print(key, value)
    
    for i in data['fields']:
        print("--------------------------------")
        for key, value in i.items():
            print(key, value)
    
    for i in data['interactions']:
        print("--------------------------------")
        for key, value in i.items():
            print(key, value)

group_key = ["id", "name", "charge", "group", "coupling", "boson"]
particle_key = ["id", "type", "name", "mass", "charge"]
field_key = ["id", "name", "type", "dim", "gen", "chirality", "self_conjugate", "reps", "QuantumNumber", "particles"]
interaction_key = ["id", "type", "fields"]

class tokenized_model:
    def __init__(self, data):
        self.data = data

    def get_group_data(self):
        return self.data['GaugeGroups']
