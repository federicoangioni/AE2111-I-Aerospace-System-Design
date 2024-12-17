import numpy as np
from variables import *
from wp4_1 import Aerodynamics, InternalForces
from wp4_2 import WingBox
from wp5_1 import SkinBuckling



wb = WingBox(c_r= c_r, c_t = None, wingspan=b, area_factor_flanges=12, intersection= intersection, tr= tr, t_spar= 0.003, t_caps= 0.002)


skin_buckling = SkinBuckling(3, wb.geometry, wb.wingspan)
print(skin_buckling.skin_AR())
