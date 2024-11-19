import numpy as np
import subprocess
from class_2_weight import OEW_est, update_OEW
from class_1_weight import predict
from variables import MTOW
from SARoptimization import optimized_AR, optimized_e, optimized_S


files_tobe_Updated = [
    "class_1_weight.py", "MatchingDiagramPlot.py", "planform.py", "SARoptimization.py",
    "CG_location.py", "fuselage.py", "class_2_weight.py", "empennage_planform.py", "class_2_drag.py"
]

Weights = []

for i in range(20):
    print(f"Current OEW is: {OEW_est}, current MTOW is {MTOW}")
    MTOW = predict(OEW_est)
    Weights.append(MTOW)

    # Update CG_location.py with the new OEW_mass_fraction
    with open("SARoptimization.py", 'r') as file:
        contents3 = file.readlines()

    with open("SARoptimization.py", 'w') as file:
        for line in contents3:
            if line.startswith("mtow = "):
                file.write(f"mtow = {MTOW}\n")
            else:
                file.write(line)

    # Update variables.py with the new MTOW, AR, S, and e values in one pass
    with open("variables.py", 'r') as file:
        lines = file.readlines()
    with open("variables.py", 'w') as file:
        for line in lines:
            if line.startswith("MTOW = "):
                file.write(f"MTOW = {MTOW}\n")
            elif line.startswith("AR = "):
                file.write(f"AR = {optimized_AR}\n")
            elif line.startswith("S = "):
                file.write(f"S = {optimized_S}\n")
            elif line.startswith("e = "):
                file.write(f"e = {optimized_e}\n")
            else:
                file.write(line)

    # Update CG_location.py with the new OEW_mass_fraction
    with open("CG_location.py", 'r') as file:
        contents2 = file.readlines()
    
    with open("CG_location.py", 'w') as file:
        for line in contents2:
            if line.startswith("OEW_mass_fraction = "):
                file.write(f"OEW_mass_fraction = {OEW_est / MTOW}\n")
            else:
                file.write(line)
    
    # Run each file with the updated parameters
    for filename in files_tobe_Updated:
        subprocess.run(["python", filename])

    OEW_est = update_OEW(MTOW)

    # Here you could optionally reload or update OEW_est if it needs to be recalculated.

print(Weights)
