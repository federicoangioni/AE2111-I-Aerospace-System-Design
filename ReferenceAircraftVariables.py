import numpy as np
import math

cruise_h = 10000 # m
landing_field_distance = 1496 # m
takeoff_field_length = 1296 # m
V_cr = .77 # Cruise speed
rhostd = 1.225
climb_rate_requirement = 6.8 # TBD
# minimum stall speed
obstacle_height = 11

C_lmax_landing = 2.5 #1.8 < Cl < 2.8 TBD
approach_speed = 68 # m/s TBD
landing_mass_fraction = 0.87 # TBD 0.88 final value
cruise_mass_fraction = 0.95 # TBD
climb_mass_fraction = 0.95
# stall/takeoff/cclimb/

C_lmax_takeoff = 1.8 # 1.6 < Cl < 2.2   TBD
C_lmax_cruise = 1.4 # 1.2 < Cl <1.8     TBD

# -- Drag Polar sheet --
AR = 11 # TBD
S_wet_S_wing = 7.6  # TBD

c_f = 0.00275 # to estimate from figure 6.3 TBD

C_D0 = c_f * S_wet_S_wing # zero-lift drag coeffficient

# assignment 6.3
psi = 0.0075 #fixed
phi = 0.97 # fixed

e = 1/(math.pi * AR * psi + 1/phi) # assumed oswald efficiency factor
B = 9 # Bypass ratio 5 < B < 15


TSFC = 22*B**(-0.19) # [g/kN/s] Thrust-specific fuel consumption of engines chosen
engine_efficiency = V_cr / (TSFC*1e-6*43)   # 43 MJ/kg specific energy for kerosene
max_flap = 35
takeoff_flap = 15
flap_angles = np.array([0, 0, 15, 15, 35, 35])
landing_gear_drag = 0.018 # 0.01 < landing_gear_drag < 0.025 TBD

# 0:cruise retracted, 1:cruise extended, 2:TO retracted, 3:TO extended, 4:landing retracted, 5:landing extended
CD_0_phases = np.array([C_D0 + 0.0013* flap_angles[0], C_D0 + 0.0013* flap_angles[1] + landing_gear_drag, C_D0 + 0.0013* flap_angles[2], 
                          C_D0 + 0.0013* flap_angles[3] + landing_gear_drag, C_D0 + 0.0013* flap_angles[4], C_D0 + 0.0013* flap_angles[5] + landing_gear_drag]) 

oswald_phases = np.array([e + degree*0.0026 for degree in flap_angles])

# Climb gradient mass fraction requirements
#print(CD_0_phases)
mass_fraction_119 = 1
mass_fraction_121a = 1
mass_fraction_121b = 1
mass_fraction_121c = 1
mass_fraction_121d = landing_mass_fraction

climb_gradient_119 = 0.032
climb_gradient_121a = 0.0
climb_gradient_121b = 0.024
climb_gradient_121c = 0.012
climb_gradient_121d = 0.021