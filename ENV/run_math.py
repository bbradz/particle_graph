import subprocess
import os

def run_math(command):
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    run_math("math -script run_math.py")
