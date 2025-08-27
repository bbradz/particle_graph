from Token2Model.model import Model
from CalcObs.observables import ObservableCalc

MODEL_NAME = "Test New Model A"
AUTHOR = "Cooper"
JSON_PATH = "/users/qniu3/physics/RL_builder-2.0/SM_test.json"
MODEL_BASE_PATH = "/users/qniu3/physics/RL_builder-2.0/Models"
OBS_LIST_PATH = "/users/qniu3/physics/RL_builder-2.0/obs_list.json"
SARAH_PATH = "/users/qniu3/physics/SARAH-4.15.4"
SPHENO_PATH = "/users/qniu3/physics/SPheno-4.0.5"


model = Model(MODEL_NAME, AUTHOR, JSON_PATH, MODEL_BASE_PATH, simplify_checklist = True)
print(model.model_name)
print(model.model_symbol)
print(model.score)
model.write_model()
model.write_checklist()


#module load mathematica 
sarah = ObservableCalc(model.model_symbol, 
                       model_base = MODEL_BASE_PATH, 
                       obs_list_path = OBS_LIST_PATH, 
                       keep_log = True, 
                       sarah_path = SARAH_PATH, 
                       spheno_path = SPHENO_PATH,
                       sigma_threshold = 2,
                       include_tachyon = False
                       )

sarah.run_sarah()
sarah.compile_spheno()
sarah.run_spheno()
sarah.minimize_chi2()
chi2_result = sarah.chi2_result
print(chi2_result)
sarah.make_plot()