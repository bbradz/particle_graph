
### ============================ ###
###       External Parameter     ###
### ============================ ###
class ExtParam:
    def __init__(self, name, BLOCKNAME, OrderBlock, Value, Description):
        self.name = name
        self.BLOCKNAME = BLOCKNAME
        self.OrderBlock = OrderBlock
        self.Value = Value
        self.Description = Description

    def __str__(self):
        return f"{self.BLOCKNAME} {self.OrderBlock} {self.Value} {self.Description}"

    def __dict__(self):
        return {
            "ParameterType": "External",
            "BLOCKNAME": self.BLOCKNAME,
            "OrderBlock": self.OrderBlock,
            "Value": self.Value,
            "Description": self.Description
        }

    def to_fr(self):
        param_entry = f"  {self.name} == {{\n"
        param_info = []
        for key, value in self.__dict__().items():
            param_info.append(f"{key:20} -> {value}")
        param_entry += "    " + ",\n    ".join(param_info) + "\n"
        param_entry += "  },\n"
        return param_entry


### ============================= ###
###       Internal Parameter      ###
### ============================= ###
class IntParam:
    def __init__(self, name, indices, definition, value, InteractionOrder, ParameterName, TeX, Description):
        self.name = name
        self.indices = indices
        self.definition = definition
        self.value = value
        self.InteractionOrder = InteractionOrder
        self.ParameterName = ParameterName
        self.TeX = TeX
        self.Description = Description

    def __str__(self):
        return f"{self.name} = {self.definition} {self.value} {self.InteractionOrder} {self.ParameterName} {self.TeX} {self.Description}"

    def __dict__(self):
        return {
            "ParameterType": "Internal",
            "Indices": self.indices,
            "Definition": self.definition,
            "Value": self.value,
            "InteractionOrder": self.InteractionOrder,
            "ParameterName": self.ParameterName,
            "TeX": self.TeX,
            "Description": self.Description
        }

    def to_fr(self):
        param_entry = f"  {self.name} == {{\n"
        param_info = []
        for key, value in self.__dict__().items():
            param_info.append(f"{key:20} -> {value}")
        param_entry += "    " + ",\n    ".join(param_info) + "\n"
        param_entry += "  },\n"
        return param_entry
