from wp4_2 import WingBox
from variables import *

import numpy as np

wingbox = WingBox(0.001, c_r = c_r, c_t = None, wingspan= b, intersection= 0.10185873605947957, tr = tr)

stringers = [6, 0.9, 'L', {'base': 10e-3, 'height': 5e-3, 'thickness base': 2e-3, 'thickness height': 2e-3}]
# print(wingbox.stringer_geometry(0, stringers= stringers))


z = np.linspace(0, 1)
wingbox.
print(wingbox.polar(z))


