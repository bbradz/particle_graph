def str_input(prompt):
    return input(prompt)

def int_input(prompt):
    while True:
        try:
            value = int(input(prompt))
            return value
        except ValueError:
            print("Please enter a valid integer.")

def float_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Please enter a valid float.")

def bool_input(prompt):
    while True:
        value = input(prompt).lower()
        if value in ['y', 'yes', 'true', '1']:
            return True
        elif value in ['n', 'no', 'false', '0']:
            return False

def list_input(prompt, length):
    while True:
        try:
            values = input(prompt)
            int_list = list(map(int, values.split(',')))
            if len(int_list) == length:
                return int_list
            else:
                print(f"Please enter {length} integers.")
        except ValueError:
            print("Please enter valid integers.")

def selection_input(prompt, options):
    print(prompt)
    for index, option in enumerate(options):
        print(f"{chr(index + 97)}) {option}")  # 97 is the ASCII value for 'a'
    
    while True:
        choice = input("Select an option (letter): ").lower().strip()
        if len(choice) == 1 and 'a' <= choice <= chr(96 + len(options)):
            # Convert letter to index (a=0, b=1, etc.)
            index = ord(choice) - 97
            return options[index]
        else:
            print(f"Please select a letter between 'a' and '{chr(96 + len(options))}'.")

def get_input(prompt, type, length=None, options=None):
    if type == 'str':
        return str_input(prompt)
    elif type == 'int':
        return int_input(prompt)
    elif type == 'float':
        return float_input(prompt)
    elif type == 'bool':
        return bool_input(prompt)
    elif type == 'list':
        return list_input(prompt, length)
    elif type == 'selection':
        if options is None:
            raise ValueError("Options must be provided for selection input")
        return selection_input(prompt, options)


class object:
    def __init__(self, string, integer, list, boolean, selection):
        self.string = string
        self.integer = integer
        self.list = list
        self.boolean = boolean
        self.selection = selection

    def __str__(self):
        return f"String: {self.string}, Integer: {self.integer}, List: {self.list}, Boolean: {self.boolean}, Selection: {self.selection}"

# Example usage
if __name__ == "__main__":
    options = ["Option 1", "Option 2", "Option 3"]
    
    user_selection = get_input("selection: ", 'selection', None, options)
    print(user_selection)
    print(type(user_selection))