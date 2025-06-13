import os
import subprocess
import numpy as np
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union


class Verifier:
    def __init__(
        self, 
        model_name: str,
        collision_process: str,
        QED_order: int,
        QCD_order: int,
        free_param: Dict[str, int],
        nevents: int,
        ebeam1: int,
        ebeam2: int,
        model_path: str, 
        mg_path: str, 
        output_path: str,
        debug: bool = False
    ):
        self.model_name = model_name
        self.collision_process = collision_process
        self.QED_order = QED_order
        self.QCD_order = QCD_order
        self.free_param = free_param
        self.nevents = nevents
        self.ebeam1 = ebeam1
        self.ebeam2 = ebeam2
        self.model_path = model_path
        self.mg_path = mg_path
        self.output_path = output_path
        self.debug = debug


    def run_mg5(self):
        pass