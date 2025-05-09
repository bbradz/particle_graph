import random, json, hashlib, os

# read json file
def read_json(JSON_PATH):
    try:
        with open(JSON_PATH, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        print(f"Error: {JSON_PATH} is not a valid JSON file.")
        return None
    except FileNotFoundError:
        print(f"Error: File {JSON_PATH} not found.")
        return None

pdg = read_json(os.path.join(os.path.dirname(__file__), "pdg.json"))
def get_pdg(mass, charge, color, spin, flavor, PDG_DATA = pdg, mass_tol = 1e-2):
    """
    Get the particle from the PDG data
    """
    candidates = PDG_DATA['particles']

    if color is not None:
        candidates = [p for p in candidates if p['color'] == color]
    if spin is not None:
        candidates = [p for p in candidates if p['spin'] == spin]
    if flavor is not None:
        candidates = [p for p in candidates if p['flavor'] == flavor]
    if charge is not None:
        candidates = [p for p in candidates if p['charge'] == charge]    
    if mass is not None:
        candidates = [p for p in candidates if abs(p['mass'] - mass) <= mass_tol * mass or (p['mass'] == mass)]
    
    if len(candidates) == 1:
            return candidates[0]
    else:
        return None


def BSM_id(mass, charge, color, spin, flavor, offset = 9000000):
    """
    Get the BSM particle ID from the PDG data
    """
    props = f"{mass:.6f}|{charge}|{color}|{spin}|{flavor}"
    hashed = hashlib.md5(props.encode()).hexdigest()
    numeric_hash = int(hashed[:8], 16)
    bsm_pdg_id = offset + (numeric_hash % (9999999 - offset))

    return bsm_pdg_id


# generate random inputs
def random_inputs(count=6):
    """Generate a list of random values that can be strings, integers, floats, or None."""
    result = []
    for _ in range(count):
        value_type = random.choice(['str', 'int'])
        
        if value_type == 'str':
            # Generate a random string of length 3-8
            length = random.randint(3, 8)
            random_string = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz_') for _ in range(length))
            result.append(random_string)
        elif value_type == 'int':
            result.append(random.randint(-100, 100))
        elif value_type == 'float':
            result.append(random.uniform(-100.0, 100.0))
        else:  # None
            result.append(None)
            
    return result

def count_calls(func):
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        return func(*args, **kwargs)
    wrapper.call_count = 0
    return wrapper

if __name__ == "__main__":
    print(random_inputs())