import numpy as np
import matplotlib.pyplot as plt
import math
import Formulas
from Formulas import climbRateTTW

def minimum_speed(wing_loadings, C_lmax_landing, approach_speed, landing_mass_fraction, rhostd):
    min_speed = np.full(len(wing_loadings), Formulas.stallSpeedWingLoading(C_lmax_landing, approach_speed, landing_mass_fraction, rhostd))
    return min_speed

def RefDesignPoint(wing_loadings):
    for i in range(1,90):
        if minimum_speed(   )[1] <= wing_loadings[i] and minimum_speed(     )[1] >= wing_loadings[i - 1]:
            rx1 = wing_loadings[i - 1]
            rx2 = wing_loadings[i]
            y1 = i-1
            y2 = i
            break
        else:
            continue

    dp = np.interp(minimum_speed(   )[1], [rx1, rx2], [climbRateTTW(   )[y1], climbRateTTW(    )[y2]])
    return dp