import pandas as pd
import numpy as np
import subprocess
import os

# os.chdir("/notebooks/Scripts")

args_list = [
    ("xgb", "yes", "local", 3, 1),
    ("lgbm", "yes", "local", 3, 100),
    ("gbr", "yes", "local", 3, 100),
    ("et", "yes", "local", 3, 100),
    ("bag", "yes", "local", 3, 100),
    ("rf", "yes", "local", 3, 100),
    ("xgb", "no", "local", 3, 100),
    ("lgbm", "no", "local", 3, 100),
    ("gbr", "no", "local", 3, 100),
    ("et", "no", "local", 3, 100),
    ("bag", "no", "local", 3, 100),
    ("rf", "no", "local", 3, 100),
    # ('svc', 'no', 'paperspace', 3, 100)
]

for args in args_list:
    subprocess.run(
        "python run_sweep.py "
        + str(args[0])
        + " "
        + str(args[1])
        + " "
        + str(args[2])
        + " "
        + str(args[3])
        + " "
        + str(args[4]),
        shell=True,
    )
