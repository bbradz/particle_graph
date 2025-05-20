import sys
import os
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

# Add the parent directory to sys.path to allow imports from sibling packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now you can import from json2fr
from json2fr.model import Model

FeynRulesPath = "/Users/cooperniu/Documents/codes/feynrules"
ModelPath = "/Users/cooperniu/Documents/codes/particle_graph/particle_graph/Models"

def run_fr(model_name):
    session = WolframLanguageSession()
    try:
        session.evaluate(wlexpr(f'$FeynRulesPath=SetDirectory["{FeynRulesPath}"]'))
        session.evaluate(wlexpr('<<FeynRules`'))
        session.evaluate(wlexpr(f'SetDirectory[$FeynRulesPath <> "/Models/{model_name}"]'))
        session.evaluate(wlexpr(f'LoadModel["{model_name}.fr"]'))
        print(f"Model {model_name} loaded successfully.")
    except Exception as e:
        print(e)
    finally:
        session.terminate()

def CheckHermiticity(session, Lagrangian):
    result = session.evaluate(wlexpr(f'CheckHermiticity[{Lagrangian}]'))
    return result

def CheckMassSpectrum(session, Lagrangian):
    result = session.evaluate(wlexpr(f'CheckMassSpectrum[{Lagrangian}]'))
    return result

def CheckKineticTermNormalisation(session, Lagrangian):
    result = session.evaluate(wlexpr(f'CheckKineticTermNormalisation[{Lagrangian}]'))
    return result


if __name__ == "__main__":
    # Example of using the Model class
    # model = Model("SM")
    # print(model)
    
    run_fr("SM")