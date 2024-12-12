import numpy as np
from variables import *
from wp4_1 import Aerodynamics, InternalForces
from wp4_2 import WingBox
from wp5_1 import SkinBuckling


print(np.sign(-5))
wb = WingBox(c_r= c_r, c_t = None, wingspan=b, area_factor_flanges=12, intersection= intersection, tr= tr, t_spar= 0.003, t_caps= 0.002)


SkinBuckling.PlotSkinAR(5, 26.9, WingBox.geometry, SkinBuckling.SkinAspectRatio)

