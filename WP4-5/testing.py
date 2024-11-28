from wp4_2 import WingBox
import numpy as np
wingbox = WingBox(5, 3.3, c_t = None, stringers = 0, tr= 0.334)

z = np.linspace(0, 1)

wingbox.Jplots(z)