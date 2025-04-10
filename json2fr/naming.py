def number_to_alphabet(number):
    if number < 1 or number > 26:
        number = (number - 1) % 26 + 1

    return chr(number + 96)  # Convert 1 to 'a', 2 to 'b', etc.

def number_to_greek(number):
    if number < 1 or number > 24:
        number = (number - 1) % 24 + 1  # Convert any int to an int between 1 and 24
    greek_letters = [
        'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa',
        'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho', 'sigma', 'tau', 'upsilon',
        'phi', 'chi', 'psi', 'omega'
    ]
    if 0 <= number <= 24:
        return greek_letters[number - 1]  # Convert 1 to 'alpha', 2 to 'beta', etc.
    else:
        raise ValueError("Number must be between 1 and 24")
    
def number_to_english(number):
    english_numbers = [
        "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
        "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen",
        "Eighteen", "Nineteen", "Twenty", "TwentyOne", "TwentyTwo", "TwentyThree",
        "TwentyFour"
    ]
    if 1 <= number <= 24:
        return english_numbers[number - 1]
    else:
        raise ValueError("Number must be between 1 and 24")

def number_to_plural(number):
    number = int(number)
    plural = ["singlet", "doublet", "triplet", "quadruplet", "quintuplet", "sextuplet", "septuplet", "octuplet", "nonuplet", "decuplet"]
    
    if number < 1 or number > 10:
        raise ValueError("Number must be between 1 and 10")
    return plural[number - 1]
    