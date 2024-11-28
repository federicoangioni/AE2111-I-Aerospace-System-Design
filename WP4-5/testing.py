from wp4_2 import WingBox
from variables import *

import numpy as np

wingbox = WingBox(0.001, c_r = c_r, c_t = None, wingspan= b, intersection= 0.102, tr = tr)




z = np.linspace(0, 1)

wingbox.Jplots(z)