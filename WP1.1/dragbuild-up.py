from variables import S, AR, LE_sweep, C4_sweep, V_cr, c_r, c_t, b, MTOW
from planform import V_inf
import numpy as np
# Values from running SARoptimization.py

## Fuselage values 
d_fus = 2.736
L_fus = 29.542
M = .77
tr = 0.235

# ISA values at 35000 ft
dyn_visc = 0.0000144446
rho = 0.379597

# 0.05% of the fuselage is laminar
laminar_fus = 0.05

k = 0.634e-5

Re = min(V_inf * L_fus*rho/dyn_visc, 44.62*((L_fus/k)**1.053) * M**1.16)

# Laminar friction coefficient
C_f_lamFus = 1.328/np.sqrt(Re)

# Turbolent friction coefficient
C_f_turbFus = 0.455/((np.log10(Re)**(2.58))*(1+0.144*M**2)**0.65)

# Total friction coefficient weighted
C_fFus = laminar_fus * C_f_lamFus + (1-laminar_fus) * C_f_turbFus

f_fus = L_fus/np.sqrt(d_fus**2)
FF_fus = (1+ 60/f_fus**3 + f_fus/ 400)

IF_fus = 1.0

L1_fus = 5.472
L2_fus = 14.494
L3_fus = 9.576

S_wet_fus = (np.pi * d_fus / 4) * ((1 / (3 * L1_fus**2)) * ((4 * L1_fus**2 + d_fus**2 / 4)**1.5 - d_fus**3 / 8)) - d_fus + 4 * L2_fus + 2 * np.sqrt(L3_fus**2 + d_fus**2 / 4)

multiplied_fus = C_fFus*S_wet_fus*IF_fus*FF_fus


## Wing drag component
mac = 3.06

def angle_at_xdivc(x, c, LEsweep, c_r, tr, b):
    return np.arctan(np.tan(LEsweep) - (x/c)  * 2 * (c_r/b) * (1-tr))

laminar_wing = 0.1
# SC(2)-0714
x_c_m = 0.37
ttoc = 0.14
lambda_m  = angle_at_xdivc(37, 100, np.radians(LE_sweep), c_r, tr, b)


k = 0.634e-5 # paint factor

Re = min(V_inf *mac*rho/dyn_visc, 44.62*((mac/k)**1.053) * M**1.16)


# Laminar friction coefficient
C_f_lamwing = 1.328/np.sqrt(Re)

# Turbolent friction coefficient
C_f_turbwing = 0.455/((np.log10(Re)**(2.58))*(1+0.144*M**2)**0.65)

# Total friction coefficient weighted
C_fwing = laminar_wing * C_f_lamwing + (1-laminar_wing) * C_f_turbwing

FF_wing = (1+ (0.6 / x_c_m) * ttoc + 100*ttoc**4)*(1.34*M**0.18*np.cos(lambda_m)**0.28)

IF_wing = 1.4 

S_wet_wing = 2*1.07*S

multiplied_wing = C_fwing*S_wet_wing*IF_wing*FF_wing

## Nacelle drag component

# C_fe for nacelle

Re_nacelle = min(V_inf * l_n*rho/dyn_visc, 44.62*((l_n/k)**1.053) * M**1.16)

C_f_nacelle = 0.455/((np.log10(Re_nacelle)**(2.58))*(1+0.144*M**2)**0.65)

# FF for nacelle

def fineness_factor(l_n, D_n):
    # Formula for the fineness factor of the nacelle
    FF_N = 1 + (0.35 / (l_n / D_n))
    return FF_N

FF_N = fineness_factor(l_n, D_n)


# IF

Q_s = 1.3
Q_g = 1

# S_wet

def fan_cowl_wetted_area(l_n, D_n, l_1, D_Hl, D_ef):
    # Formula for the wetted area of the fan cowl
    S_wet_fan_cowl = l_n * D_n * (2 + 0.35 * (l_1 / l_n) + 0.8 * (l_1 * D_Hl) / (l_n * D_n) + 
                     1.15 * (1 - (l_1 / l_n)) * (D_ef / D_n))
    return S_wet_fan_cowl

l_n = 1.0  # Length of the nacelle
D_n = 1.0  # Diameter of the fan cowl
l_1 = 0.5  # The distance l1 is measured from the leading edge to the position of maximum thickness of the fan cowling
D_Hl = 0.3  # Diameter of the fan
D_ef = 0.9  # Diameter of back of nacelle


def gas_generator_wetted_area(l_g, D_g, D_eg):
    # Formula for the wetted area of the gas generator
    S_wet_gas_gen = (np.pi * l_g * D_g * 
                     (1 - (1 / 3) * (1 - (D_eg / D_g)) * 
                     (1 - 0.18 * (D_g / l_g) ** (5 / 3))))
    return S_wet_gas_gen


# Example usage (replace with actual values)
l_g = 1.0  # Length of the gas generator
D_g = 1.0  # Diameter of the gas generator
D_eg = 0.5  # Effective diameter of the gas generator

S_wet_gas_gen = gas_generator_wetted_area(l_g, D_g, D_eg)
print(S_wet_gas_gen)

def plug_wetted_area(l_p, D_p):
    # Formula for the wetted area of the plug
    S_wet_plug = 0.7 * np.pi * l_p * D_p
    return S_wet_plug

# Example usage (replace with actual values)
l_p = 1.0  # Length of the plug
D_p = 0.5  # Diameter of the plug

S_wet_plug = plug_wetted_area(l_p, D_p)
print(S_wet_plug)

## Vertical 
S_emp = 0
laminar_vertemp = 0.1
mac_vert_emp = 0
x_c_mvert = 0
ttocvert = 0
lambda_mvert = 0

Re = min(V_inf * mac_vert_emp*rho/dyn_visc, 44.62*((mac_vert_emp/k)**1.053) * M**1.16)

# Laminar friction coefficient
C_f_lamvertemp = 1.328/np.sqrt(Re)

# Turbulent friction coefficient
C_f_turbvertemp =  0.455/((np.log10(Re)**(2.58))*(1+0.144*M**2)**0.65)

IF_vertemp = 1.045

C_fvertemp = laminar_vertemp * C_f_lamvertemp + (1-laminar_vertemp) * C_f_turbvertemp

FF_vert_emp = (1+ (0.6 / x_c_mvert) * ttocvert + 100*ttocvert**4)*(1.34*M**0.18*np.cos(lambda_mvert)**0.28)

S_wet_emp_vert = 2*1.05*S_emp

## Wave drag estimation

q = 0.5*rho*V_inf**2
C_Ldes = 0.56 # Value from WP1.1

M_DD = 0.87/(np.cos(C4_sweep)) - (ttoc)/((np.cos(C4_sweep)**2)) - C_Ldes/(10*((np.cos(C4_sweep)**3)))

DeltaCD_wave = 0.002*(1+2.5*(M_DD-M)/0.05)**(-1)