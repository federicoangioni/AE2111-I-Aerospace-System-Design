import numpy as np
import subprocess
from class_2_weight import MTOW, OEW_est
from class_1_weight import predict

files_tobe_Updated = ["class_1_weight.py", "MatchingDiagramPlot.py", "planform.py", "SARoptimization.py", "CG_location.py",  "fuselage.py", "class_2_weight.py",  "empennage_planform.py", "class_2_drag.py"]

OEW_i = OEW_est

for i in range(10):
    print(f"Current OEW is:{OEW_i}")
    MTOW = predict(OEW_est)#7200/(1-(OEW_i/MTOW)-(8100/MTOW))
    
    for filename in files_tobe_Updated:
        with open("variables.py", 'r') as file:
            lines = file.readlines()
        with open("variables.py", 'w') as file:   
            for line in lines:
                if line.startswith("MTOW = "):
                    file.write(f"MTOW = {MTOW}\n")
                else:
                    file.write(line)

        with open("CG_location.py", 'r') as file:
            contents2 = file.readlines()

        with open("CG_location.py", 'w') as file:   
            for line in contents2:
                if line.startswith("OEW_mass_fraction = "):
                    file.write(f"OEW_mass_fraction = {OEW_i/MTOW}\n")
                else:
                    file.write(line)
    
    # Run each file with the updated parameter
    for filename in files_tobe_Updated:
        subprocess.run(["python", filename])

    OEW_i = OEW_est