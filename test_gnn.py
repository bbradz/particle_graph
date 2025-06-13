from _EinsteinNET.EinsteinNet import create_model_and_test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

json_path = "theory_data/SM.json"
create_model_and_test(json_path)