class Parameter:
    all_parameters = {}

    def __init__(self, name: str, 
                 Description: str, 
                 OutputName: str, 
                 Block = None,
                 Dependence = None,
                 DependenceNum = None, 
                 DependenceSPheno = None,
                 DependenceOptional = None, 
                 Real = None,
                 Value = None,
                 LesHouches = None,
                 LaTeX = None):
        
        #check if the parameter already exists
        if name in Parameter.all_parameters.keys():
            raise ValueError(f"Parameter with name '{name}' already exists")
        if Description in Parameter.all_parameters.values():
            raise ValueError(f"Parameter with description '{Description}' already exists")
        Parameter.all_parameters[name] = Description
        
        #set the parameters
        self.name = name
        self.Description = Description
        self.OutputName = OutputName
        self.Block = Block
        self.Dependence = Dependence
        self.DependenceNum = DependenceNum
        self.DependenceSPheno = DependenceSPheno
        self.DependenceOptional = DependenceOptional
        self.Real = Real
        self.Value = Value
        self.LesHouches = LesHouches
        self.LaTeX = LaTeX

    def __str__(self):
        return f"{self.name} ({self.Description})"

    def __dict__(self):
        return {
            "Description": self.Description,
            "OutputName": self.OutputName,
            "Dependence": self.Dependence,
            "DependenceNum": self.DependenceNum,
            "DependenceSPheno": self.DependenceSPheno,
            "DependenceOptional": self.DependenceOptional,
            "Real": self.Real,
            "Value": self.Value,
            "LesHouches": self.LesHouches,
            "LaTeX": self.LaTeX
        }


class ExternalParameter(Parameter):
    def __init__(self, name: str, 
                 Description: str, 
                 OutputName: str, 
                 Block: str,
                 Dependence = None,
                 DependenceNum = None,
                 DependenceSPheno = None,
                 DependenceOptional = None,
                 Real = False,
                 Value = None,
                 LesHouches = None,
                 LaTeX = None):
        super().__init__(name, Description, OutputName, Block, Dependence, DependenceNum, DependenceSPheno, DependenceOptional, Real, Value, LesHouches, LaTeX)
        


class InternalParameter(Parameter):
    def __init__(self, name: str, 
                 Description: str, 
                 OutputName: str, 
                 Block: str,
                 Dependence = None,
                 DependenceNum = None, 
                 DependenceSPheno = None,
                 DependenceOptional = None, 
                 Real = False,
                 Value = None,
                 LesHouches = None,
                 LaTeX = None):
        super().__init__(name, Description, OutputName, Block, Dependence, DependenceNum, DependenceSPheno, DependenceOptional, Real, Value, LesHouches, LaTeX)



if __name__ == "__main__":
    param = Parameter(name = "g1", 
                      Description = "g1", 
                      OutputName = "g1"
                      )
    param2 = Parameter(name = "g22", 
                      Description = "g2", 
                      OutputName = "g2"
                      )