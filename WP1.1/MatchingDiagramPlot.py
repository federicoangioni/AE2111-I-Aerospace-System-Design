import matplotlib.pyplot as plt
import numpy as np
from variables import *
from Formulas import *
from Extrasheet import *


# Initialisation of x and y value lists
wing_loadings = np.arange(0, 9000, 1)

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

figure, axis = plt.subplots(figsize=(9,6))
axis.grid(True, alpha=0.9)
axis.plot(minimum_speed, wing_loadings, label= "Minimum Speed", color="blue")
axis.plot(lfl, wing_loadings, label = "Landing Field Length", color = "orange")
axis.plot(wing_loadings[1:], CruiseSpeed[1:], label = "Cruise Speed", color = "purple")
axis.plot(wing_loadings[1:], Climbrate[1:], label="Climb rate")
axis.plot(wing_loadings[1:], ClimbG121a[1:], label="Climb gradient CS25.121a")
axis.plot(wing_loadings[1:], ClimbG121b[1:], label="Climb gradient CS25.121b")
axis.plot(wing_loadings[1:], ClimbG121c[1:], label="Climb gradient CS25.121c")
axis.plot(wing_loadings[1:], ClimbG121d[1:], label="Climb gradient CS25.121d")
axis.plot(wing_loadings[1:], ClimbG119[1:], label="Climb gradient CS25.119")
axis.plot(wing_loadings[1:], TakeOffLength[1:], label = "Take off Length")

for i in range(1,len(wing_loadings)):
    if minimum_speed[1] <= wing_loadings[i] and minimum_speed[1] >= wing_loadings[i - 1]:
        rx1 = wing_loadings[i - 1]
        rx2 = wing_loadings[i]
        y1 = i-1
        y2 = i
        break
    else:
        continue
dp = np.interp(minimum_speed[1], [rx1, rx2], [Climbrate[y1], Climbrate[y2]])
plt.scatter(minimum_speed[2], dp, label = "Design Point", color = "yellow", marker = "D", zorder = 1000, edgecolors="black", s=60, linewidths=1.5)

wing_surface = MTOW*9.81/minimum_speed[2]

plt.annotate(f'({round(minimum_speed[1])}, {round(dp, 2)})',
             xy=(minimum_speed[1], dp),
             xytext=(minimum_speed[1] - 770 , dp + 0.12),
             arrowprops=dict(arrowstyle='->', lw=1.5))

tw = [0.379, 0.324, 0.373, 0.347, 0.304, 0.287, 0.382, 0.361, 0.295, 0.315],
 
ws = [4585.272, 5208.611, 4882.186, 4862.099, 3785.325, 4740.997, 4975.019, 5332.789, 5788.837, 4253.758]
plt.scatter(ws, tw, marker= "o", color="black", label= "Reference Aircraft", s=15, zorder=999)

choice = np.array([])
for value in range(len(Climbrate)):
    choice = np.append(choice, max(CruiseSpeed[value], Climbrate[value]))

axis.fill_between(wing_loadings[:5282], y1= choice[:5282], y2= minimum_speed[:5282],color="green", alpha = 0.2)

plt.xlabel(r'Wing loading  $W_{TO}/{S_w} \ [N/m^2]$', fontsize = 10)
plt.ylabel(r'Thrust-to-weight ratio  ${T_{TO}/{W_{TO}}} \ [N/N]$', fontsize=10)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          fancybox=True, shadow=True, ncol=3)
plt.xlim(0, 6500)
plt.ylim(0, 0.5)
plt.tight_layout()
if __name__ == "__main__":
    plt.show()
