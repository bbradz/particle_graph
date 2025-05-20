from utility import count_calls

@count_calls
def generate_id():
    return f"id-{generate_id.call_count}"

@count_calls
def generate_dummy_idx():
    return f"{num2iii(generate_dummy_idx.call_count)}"

@count_calls
def generate_ijk():
    return f"{num2ijk(generate_ijk.call_count)}"

def num2ijk(num):
    if num <= 0:
        return ""
    num = num + 9
    letter = chr(ord('a') + (num - 1) % 26)
    repeat = (num - 1) // 26 + 1
    
    return letter * repeat

def num2iii(num):
    """
    Convert a number to a sequence of letters where:
    """
    if num <= 0:
        return ""
    num = num + 26
    letter = chr(ord('a') + (num - 1) % 26)
    repeat = (num - 1) // 26 + 1
    
    return letter * repeat

def num2words(num):
    """
    Convert a number to its English words.
    Only handles numbers less than or equal to 100.
    """
    if num < 0:
        return "minus " + num2words(abs(num))
    
    ones = {
        0: 'zero',
        1: 'one',
        2: 'two',
        3: 'three',
        4: 'four',
        5: 'five',
        6: 'six',
        7: 'seven',
        8: 'eight',
        9: 'nine',
        10: 'ten',
        11: 'eleven',
        12: 'twelve',
        13: 'thirteen',
        14: 'fourteen',
        15: 'fifteen',
        16: 'sixteen',
        17: 'seventeen',
        18: 'eighteen',
        19: 'nineteen'
    }
    
    tens = {
        2: 'twenty',
        3: 'thirty',
        4: 'forty',
        5: 'fifty',
        6: 'sixty',
        7: 'seventy',
        8: 'eighty',
        9: 'ninety'
    }
    if num > 100:
        return num2words(num % 100)
    elif num < 20:
        return ones.get(num, '')
    elif num < 100:
        return tens.get(num // 10, '') + ('-' + ones.get(num % 10, '') if num % 10 != 0 else '')
    else:
        return "one hundred"

def num2abc(number):
    """
    Convert a number to Excel-style column letters.
    """
    if number <= 0:
        raise ValueError("Number must be positive")
    
    result = ""
    while number > 0:
        remainder = (number - 1) % 26
        result = chr(remainder + 97) + result  # 97 is ASCII for 'a'
        number = (number - 1) // 26
    
    return result

def num2greek(number):
    """
    Convert a number to its corresponding Greek letter name.
    Examples: 1→alpha, 2→beta, etc.
    """
    if number <= 0:
        number = abs(number)
    
    # Modulo to handle numbers greater than 24
    number = ((number - 1) % 24) + 1
    
    greek_letters = {
        1: 'alpha', 
        2: 'beta', 
        3: 'gamma', 
        4: 'delta',
        5: 'epsilon',
        6: 'zeta',
        7: 'eta',
        8: 'theta',
        9: 'iota',
        10: 'kappa',
        11: 'lambda',
        12: 'mu',
        13: 'nu',
        14: 'xi',
        15: 'omicron',
        16: 'pi',
        17: 'rho',
        18: 'sigma',
        19: 'tau',
        20: 'upsilon',
        21: 'phi',
        22: 'chi',
        23: 'psi',
        24: 'omega'
    }
    
    return greek_letters[number]

def num2tuple(number):
    """
    Convert a number to its corresponding tuple name (singlet, doublet, triplet, etc.)
    """
    if number <= 0:
        number = abs(number)
    
    tuple_names = {
        1: 'singlet',
        2: 'doublet',
        3: 'triplet',
        4: 'quartet',
        5: 'quintet',
        6: 'sextet',
        7: 'septet',
        8: 'octet',
        9: 'nonet',
        10: 'decuplet'
    }
    
    if number in tuple_names:
        return tuple_names[number]
    else:
        return f"{num2words(number)}-plet"


def fermion_field_name(charge, color):
    if color == 1:
        if charge == 0:
            return "vl", "neutrino"
        elif charge == -3:
            return "l", "charged lepton"
        else:
            symbol = f"{num2words(charge)} charged lepton"
            return f"{symbol[0]}l", symbol
    else:
        if charge == 2:
            return "uq", "up quark"
        elif charge == -1:
            return "dq", "down quark"
        else:
            symbol = f"{num2words(charge)} quark"
            return f"{symbol[0]}q", symbol

# Example usage
if __name__ == "__main__":
    # for i in range(1, 10):
    #     print(f"{i} -> {num2words(i)}")
    #     print(f"{i} -> {num2abc(i)}")
    #     print(f"{i} -> {num2greek(i)}")
    #     print(f"{i} -> {num2tuple(i)}")
    #     print()

    for i in range(100):
        print(generate_dummy_idx())