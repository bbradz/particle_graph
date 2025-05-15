from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

FeynRulesPath = "/Users/cooperniu/Documents/codes/feynrules"
ModelPath = "/Users/cooperniu/Documents/codes/particle_graph/particle_graph/SM"

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
    run_fr("SM")