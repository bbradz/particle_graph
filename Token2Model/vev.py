class vev:
    def __init__(self, id, name, vacuum, value):
        self.id = id
        self.name = name
        self.vacuum = vacuum
        self.value = value
        self._input_validate()

    def _input_validate(self):
        id_check = isinstance(self.id, str)
        name_check = isinstance(self.name, str)
        vacuum_check = isinstance(self.vacuum, list)
        value_check = isinstance(self.value, int) or isinstance(self.value, float)

        self.checklist = {"id": id_check, 
                          "name": name_check,
                          "vacuum": vacuum_check,
                          "value": value_check}
        
        max_score = len(self.checklist)
        score = sum(1 for value in self.checklist.values() if value is True)
        self.score = f"{score}/{max_score}"
        self.errors = {key: value for key, value in self.checklist.items() if value is not True}

    @property
    def nonzero_idx(self):
        import numpy as np
        return np.nonzero(self.vacuum)[0].tolist()