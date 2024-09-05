import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from variables import *
from Formulas import *
from Extrasheet import *

# Initialisation of x and y value lists
wing_loadings = np.arange(0, 9000, 100)

minimum_speed = np.full(len(wing_loadings), stallSpeedWingLoading(C_lmax_landing, approach_speed, landing_mass_fraction, rhostd))
lfl = np.full(len(wing_loadings), LandingFieldLength(C_lmax_landing, landing_field_distance, landing_mass_fraction, Clfl=0.45, density=rhostd))
CruiseSpeed = np.array([0])
Climbrate = np.array([0])
ClimbG119 = np.array([0])
ClimbG121a = np.array([0])
ClimbG121b = np.array([0])
ClimbG121c = np.array([0])
ClimbG121d = np.array([0])
TakeOffLength = np.array([0])

for load in wing_loadings[1:]:
    CruiseSpeed = np.append(CruiseSpeed, cruiseTTW(cruise_mass_fraction, CD_0_phases[0], V_cr,  AR, oswald_phases[0], load, cruise_h))
    Climbrate = np.append(Climbrate, climbRateTTW(climb_rate_requirement, climb_mass_fraction, C_D0, AR, load, e))
    ClimbG119 = np.append(ClimbG119, climbGradientTTW(mass_fraction_119, climb_gradient_119, load, CD_0_phases[5], AR, oswald_phases[5]))
    ClimbG121a = np.append(ClimbG121a, climbGradientTTW(mass_fraction_121a, climb_gradient_121a, load, CD_0_phases[3], AR, oswald_phases[3], OEI=True))
    ClimbG121b = np.append(ClimbG121b, climbGradientTTW(mass_fraction_121b, climb_gradient_121b, load, CD_0_phases[2], AR, oswald_phases[2], OEI=True))
    ClimbG121c = np.append(ClimbG121c, climbGradientTTW(mass_fraction_121c, climb_gradient_121c, load, CD_0_phases[0], AR, oswald_phases[0], OEI=True))
    ClimbG121d = np.append(ClimbG121d, climbGradientTTW(mass_fraction_121d, climb_gradient_121d, load, CD_0_phases[4], AR, oswald_phases[4], OEI=True))
    TakeOffLength = np.append(TakeOffLength, takeOffFieldLength(load, takeoff_field_length, 1.225, AR, oswald_phases[3], obstacle_height))


figure, axis = plt.subplots(figsize=(11,5))
axis.grid(True, alpha=0.9)
axis.plot(minimum_speed, wing_loadings, label= "Minimum Speed", color="blue")
axis.plot(lfl, wing_loadings, label = "Landing Field Length", color = "orange")
axis.plot(wing_loadings, CruiseSpeed, label = "Cruise Speed", color = "purple")
axis.plot(wing_loadings[1:], Climbrate[1:], label="Climb rate")
axis.plot(wing_loadings[1:], ClimbG121a[1:], label="Climb gradient CS25.121a")
axis.plot(wing_loadings[1:], ClimbG121b[1:], label="Climb gradient CS25.121b")
axis.plot(wing_loadings[1:], ClimbG121c[1:], label="Climb gradient CS25.121c")
axis.plot(wing_loadings[1:], ClimbG121d[1:], label="Climb gradient CS25.121d")
axis.plot(wing_loadings[1:], ClimbG119[1:], label="Climb gradient CS25.119")
axis.plot(wing_loadings[1:], TakeOffLength[1:], label = "Take off Length")

plt.legend(bbox_to_anchor=(1.025, 1.0), loc='upper left')
plt.xlim(0, 6000)
plt.ylim(0, 0.5)
plt.tight_layout()



def Design_Point(climb, speed):
    diff = np.abs(climb[10:] - speed[10:])  
    print(climb, speed)
Design_Point(Climbrate, minimum_speed)
plt.show()