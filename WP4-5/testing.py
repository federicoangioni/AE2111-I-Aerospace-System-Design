import numpy as np
from variables import *
from wp4_1 import Aerodynamics, InternalForces
from wp4_2 import WingBox
from wp5_1 import SkinBuckling
import matplotlib.pyplot as plt


wb = WingBox(c_r= c_r, c_t = None, wingspan=b, area_factor_flanges=12, intersection= intersection, tr= tr, t_spar= 0.003, t_caps= 0.002)

#print(wb.geometry(wb.wingspan/2))
skin_buckling = SkinBuckling(n_ribs=20, wingbox_geometry=wb.geometry, wingspan=wb.wingspan, E= E, v=0.33, t_skin=0.001)
#print(skin_buckling.skin_buckling_constant(aspect_ratio=5, show= True))
# skin_buckling.skin_buckling_constant(aspect_ratio=5, show= True)
skin_buckling.plot_skinAR()
skin_buckling.plot_sigma_cr()