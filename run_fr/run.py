import sys
import os
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

def run_feynrules(session, FeynRulesPath, model_path, model_name):
    try:
        session.evaluate(wlexpr(f'$FeynRulesPath=SetDirectory["{FeynRulesPath}"]'))
        session.evaluate(wlexpr('<<FeynRules`'))
        session.evaluate(wlexpr(f'SetDirectory[{model_path}]'))
        session.evaluate(wlexpr(f'LoadModel["{model_name}.fr"]'))
        print(f"Model {model_name} loaded successfully.")
    except Exception as e:
        print(f"Error loading model {model_name}:")
        print(e)

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
    pass