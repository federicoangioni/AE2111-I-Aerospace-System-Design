from scipy.interpolate import interp1d
import numpy as np
from variables import *
from k_s_curve import k_s_array 


def back_spar_web_buckling(self, a_over_b, E, t_sparweb, b):
    k_s_array_np = np.array(k_s_array)
    ab_values = k_s_array_np[:, 0]
    k_s_values = k_s_array_np[:, 1]
    k_s = np.interp(a_over_b, ab_values, k_s_values)
    crit_stress = np.pi**2 * k_s * E /(12*(1-0.33**2)) * (t_sparweb/b)**2

back_spar_web_buckling(1.5, 40000, 0.05, 2)
