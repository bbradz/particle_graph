class ParameterRegistry:
    """
    Global registry to track all parameter names and prevent duplicates.
    This registry is shared across all models and can be cleared between model creations.
    """
    def __init__(self):
        self._parameters = {}  # name -> info dictionary
        self._descriptions = {}  # description -> name mapping
    
    def add_parameter(self, name: str, description: str, param_type: str = "unknown"):
        if name in self._parameters:
            raise ValueError(f"Parameter with name '{name}' already exists")
        if description in self._descriptions:
            raise ValueError(f"Parameter with description '{description}' already exists")
        
        self._parameters[name] = {
            'description': description,
            'type': param_type
        }
        self._descriptions[description] = name
    
    def exists(self, name: str) -> bool:
        """Check if a parameter name already exists in the registry."""
        return name in self._parameters
    
    def description_exists(self, description: str) -> bool:
        """Check if a parameter description already exists in the registry."""
        return description in self._descriptions
    
    def get_parameter_info(self, name: str) -> dict:
        """Get information about a parameter."""
        return self._parameters.get(name, {})
    
    def all_parameters(self) -> list:
        """Get list of all registered parameter names."""
        return list(self._parameters.keys())
    
    def get_by_type(self, param_type: str) -> list:
        """Get all parameter names of a specific type."""
        return [name for name, info in self._parameters.items() if info['type'] == param_type]
    
    def clear(self):
        """Clear all registered parameters. Use this before creating a new model."""
        self._parameters.clear()
        self._descriptions.clear()
        # Force garbage collection to ensure references are broken
        import gc
        gc.collect()
        
    def __len__(self):
        return len(self._parameters)
    
    def __str__(self):
        return f"ParameterRegistry with {len(self._parameters)} parameters"


# Global registry instance
GlobalParameterRegistry = ParameterRegistry()


class Parameter:
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
                 LaTeX = None,
                 param_type: str = "parameter"):
        
        #check if the parameter already exists using the global registry
        if GlobalParameterRegistry.exists(name):
            raise ValueError(f"Parameter with name '{name}' already exists")
        if GlobalParameterRegistry.description_exists(Description):
            raise ValueError(f"Parameter with description '{Description}' already exists")
        
        # Add to both the global registry and the class-level tracking
        GlobalParameterRegistry.add_parameter(name, Description, param_type)
        
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
        super().__init__(name, Description, OutputName, Block, Dependence, DependenceNum, DependenceSPheno, DependenceOptional, Real, Value, LesHouches, LaTeX, param_type="external")
        


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
        super().__init__(name, Description, OutputName, Block, Dependence, DependenceNum, DependenceSPheno, DependenceOptional, Real, Value, LesHouches, LaTeX, param_type="internal")



if __name__ == "__main__":
    param = Parameter(name = "g1", 
                      Description = "g1", 
                      OutputName = "g1"
                      )
    param2 = Parameter(name = "g22", 
                      Description = "g2", 
                      OutputName = "g2"
                      )