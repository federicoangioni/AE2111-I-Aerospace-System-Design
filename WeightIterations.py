import numpy as np
import subprocess
from class_2_weight import MTOW, OEW_est

files_tobe_Updated = ["/CG_location.py", "/class_1_weight.py", "/class_2_drag.py", "/class_2_weight.py", "/empennage_planform.py", "/fuselage.py", "/MatchingDiagramPlot.py", "/planform.py", "/SARoptimization.py"]

Fuel_mass_fraction = 8100/MTOW

OEW_i = OEW_est

for i in range(100):
    print(f"Current OEW is:{OEW_i}")
    MTOW = 7200/(1-(OEW_i/MTOW)-(Fuel_mass_fraction/MTOW))
    OEW_i = OEW_est
    for filename in files_tobe_Updated:
        with open("/variables.py", 'r') as file:
            contents = file.read()

        contents = contents.replace(f"MTOW = {MTOW}", f"MTOW = {MTOW}")
        
        with open(filename, 'w') as file:   
            file.write(contents)
    
    # Run each file with the updated parameter
    for filename in files_to_run:
        subprocess.run(["python", filename])