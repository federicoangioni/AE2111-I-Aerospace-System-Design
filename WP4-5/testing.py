import numpy as np
from variables import *
from wp4_1 import Aerodynamics, InternalForces
from wp4_2 import WingBox
from wp5_1 import SkinBuckling


print(np.sign(-5))
wb = WingBox(c_r= c_r, c_t = None, wingspan=b, area_factor_flanges=12, intersection= intersection, tr= tr, t_spar= 0.003, t_caps= 0.002)
# ct = wb.centroid(1,20)
stringers = [8, 1, 'I', {'base': 30e-3, 'top': 30e-3, 'web height':30e-3, 'thickness top':2e-3, 'thickness web':2e-3, 'thickness base': 2e-3, }]
wb.plot_centroid(wb.z,20)

SkinBuckling.PlotSkinAR(5, 26.9, WingBox.geometry, SkinBuckling.SkinAspectRatio)

