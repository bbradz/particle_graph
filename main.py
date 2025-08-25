from Token2Model.model import Model
from CalcObs.observables import ObservableCalc

model_name = "A A New Model"
author = "Cooper"
json_path = "SM_test.json"
output_path = "Models"
obv_list_path = "obs_list.json"


model = Model(model_name, author, json_path, output_path)
print(model.model_symbol)
print(model.score)
model.write_model()
model.write_checklist()

# sarah = ObservableCalc("AANM", model_base = "Models", obv_list_path = obv_list_path, keep_log = False)
# # sarah.run_sarah()
# # sarah.compile_spheno()
# # sarah.run_spheno()
# #sarah.minimize_chi2()
# sarah.make_plot()
