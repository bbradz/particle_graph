import os 
import sys 
import json

import subprocess


FEYNMAN_PATH = "/users/qniu3/physics/FeynRules"
MADGRAPH_PATH = "/users/qniu3/physics/MG5_aMC_v3_6_2"

class Verifier:
    def __init__(self, 
                 model_path,
                 feynman_path = FEYNMAN_PATH,
                 madgraph_path = MADGRAPH_PATH):

        self.model_path = model_path
        self.feynman_path = feynman_path
        self.madgraph_path = madgraph_path

    def verify(self):
        pass
