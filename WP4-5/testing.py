import numpy as np
from variables import *
from wp4_1 import Aerodynamics, InternalForces
from wp4_2 import WingBox
from wp5_1 import SkinBuckling
from wp5_1 import Stringer_bucklin
import matplotlib.pyplot as plt

t_caps = 0.002
def M(x):
    
    return x
def I(x, y):
    return x
wb = WingBox(c_r= c_r, c_t = None, wingspan=b, area_factor_flanges=12, intersection= intersection, tr= tr, t_spar= 0.003, t_caps= t_caps)

#print(wb.geometry(wb.wingspan/2))
skin_buckling = SkinBuckling(n_ribs=12, wingbox_geometry=wb.geometry, M =M, N= M, stringers=1, I_tot=I, wingspan=wb.wingspan, E= E, v=0.33, t_skin=t_caps)
#print(skin_buckling.skin_buckling_constant(aspect_ratio=5, show= True))
skin_buckling.skin_buckling_constant(aspect_ratio=5, show= True)

(skin_buckling.applied_stress(z=1))

