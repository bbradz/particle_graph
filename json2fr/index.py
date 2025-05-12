### =============================== ###
###          Index Classes          ###
### =============================== ###
class Index:
    """
    Index class for indices of fields.
    name: str
    dim: int
    fold: str
    color: bool
    """
    def __init__(self, name, dim, fold, color = False):
        self.name = name
        self.dim = dim
        self.fold = fold
        self.color = color

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Index[{self.name}]"
    
    def IndexRange(self):
        if self.fold != "Fold":
            idx_range = f"IndexRange[Index[{self.name:8}]] = {self.fold}[Range[{self.dim}]];"
        else:
            idx_range = f"IndexRange[Index[{self.name:8}]] = Range[{self.dim}];"
        return idx_range
    
    def IndexStyle(self, style):
        return f"IndexStyle[{self.name:8}, {style}]"

    
if __name__ == "__main__":
    i = Index("ThreeGen", 3, "3")
    print(i.Index())
    print(i.IndexRange())
    print(i.IndexStyle("i"))