import numpy as np
from fuselage import fuselage_length, d_fus, l_nc, l_tc
from planform import MAC, c_r, taper_ratio, wing_surface, b, angle_at_xdivc, LESweep, c4sweep
from variables import M_cr, V_inf, l_nacelle
from empennage_planform import ARvert, ARhoriz, MACvert, SV, LEsweepvert, taperingvert, SH, MAChoriz, LEsweephoriz, taperinghoriz

# ISA values at 35000 ft
dyn_visc = 0.0000144446
rho = 0.379597
k = 0.634e-5

# ------------------------------------------------------------------------------------------------------------------
# Fuselage drag component
laminar_fus = 0.05

Re = min(V_inf * fuselage_length*rho/dyn_visc, 44.62*((fuselage_length/k)**1.053) * M_cr**1.16)

# Laminar friction coefficient
C_f_lamFus = 1.328/np.sqrt(Re)

# Turbolent friction coefficient
C_f_turbFus = 0.455/((np.log10(Re)**(2.58))*(1+0.144*M_cr**2)**0.65)

# Total friction coefficient weighted
C_fFus = laminar_fus * C_f_lamFus + (1-laminar_fus) * C_f_turbFus

f_fus = fuselage_length/np.sqrt(d_fus**2)

FF_fus = (1+ 60/f_fus**3 + f_fus/ 400)

IF_fus = 1.0

L2_fus = fuselage_length - l_nc - l_tc

S_wet_fus = (np.pi * d_fus / 4) * ((1 / (3 * l_nc**2)) * ((4 * l_nc**2 + d_fus**2 / 4)**1.5 - d_fus**3 / 8)) - d_fus + 4 * L2_fus + 2 * np.sqrt(l_tc**2 + d_fus**2 / 4)

multiplied_fus = C_fFus*S_wet_fus*IF_fus*FF_fus

# ------------------------------------------------------------------------------------------------------------------------
# Wing component
laminar_wing = 0.1
# SC(2)-0714
x_c_m = 0.37
ttoc = 0.14
lambda_m  = angle_at_xdivc(37, 100, np.radians(LESweep), c_r, taper_ratio, b)


k = 0.634e-5 # paint factor

Re = min(V_inf *MAC*rho/dyn_visc, 44.62*((MAC/k)**1.053) * M_cr**1.16)


# Laminar friction coefficient
C_f_lamwing = 1.328/np.sqrt(Re)

# Turbolent friction coefficient
C_f_turbwing = 0.455/((np.log10(Re)**(2.58))*(1+0.144*M_cr**2)**0.65)

# Total friction coefficient weighted
C_fwing = laminar_wing * C_f_lamwing + (1-laminar_wing) * C_f_turbwing

FF_wing = (1+ (0.6 / x_c_m) * ttoc + 100*ttoc**4)*(1.34*M_cr**0.18*np.cos(lambda_m)**0.28)

IF_wing = 1.4 

S_wet_wing = 2*1.07*wing_surface

multiplied_wing = C_fwing*S_wet_wing*IF_wing*FF_wing

# Wave drag estimation

q = 0.5*rho*V_inf**2
C_Ldes = 0.56 # Value from WP1.1

M_DD = 0.87/(np.cos(c4sweep)) - (ttoc)/((np.cos(c4sweep)**2)) - C_Ldes/(10*((np.cos(c4sweep)**3)))

deltaCD_wave = 0.002*(1+2.5*(M_DD-M_cr)/0.05)**(-1)
#--------------------------------------------------------------------------------------------------------------------------
# Nacelle
  # Length of the nacelle
D_n = 1.4478  # Diameter of the fan cowl
l_1 = 0.5  # The distance l1 is measured from the leading edge to the position of maximum thickness of the fan cowling
D_Hl = 1.346 # Diameter of the fan
D_ef = 1.4  # Diameter of back of nacelle

# C_fe for nacelle

Re_nacelle = min(V_inf * l_nacelle*rho/dyn_visc, 44.62*((l_nacelle/k)**1.053) * M_cr**1.16)

C_f_nacelle = 0.455/((np.log10(Re_nacelle)**(2.58))*(1+0.144*M_cr**2)**0.65)

# FF for nacelle
FF_N = 1 + (0.35 / (l_nacelle / D_n))

# IF
Q_s = 1.3
Q_g = 1

# S_wet
def fan_cowl_wetted_area(l_nacelle, D_n, l_1, D_Hl, D_ef):
    # Formula for the wetted area of the fan cowl
    S_wet_fan_cowl = l_nacelle * D_n * (2 + 0.35 * (l_1 / l_nacelle) + 0.8 * (l_1 * D_Hl) / (l_nacelle * D_n) + 
                     1.15 * (1 - (l_1 / l_nacelle)) * (D_ef / D_n))
    return S_wet_fan_cowl

S_wet_fan_cowl = fan_cowl_wetted_area(l_nacelle, D_n, l_1, D_Hl, D_ef)

def gas_generator_wetted_area(l_g, D_g, D_eg):
    # Formula for the wetted area of the gas generator
    S_wet_gas_gen = (np.pi * l_g * D_g * 
                     (1 - (1 / 3) * (1 - (D_eg / D_g)) * 
                     (1 - 0.18 * (D_g / l_g) ** (5 / 3))))
    return S_wet_gas_gen

l_g = 1.0  # Length of the gas generator
D_g = 1.0  # Diameter of the gas generator
D_eg = 0.5  # Effective diameter of the gas generator

S_wet_gas_gen = gas_generator_wetted_area(l_g, D_g, D_eg)

def plug_wetted_area(l_p, D_p):
    # Formula for the wetted area of the plug
    S_wet_plug = 0.7 * np.pi * l_p * D_p
    return S_wet_plug

l_p = 1.0  # Length of the plug
D_p = 0.5  # Diameter of the plug

S_wet_plug = plug_wetted_area(l_p, D_p)

multiplied_engine = C_f_nacelle*FF_N*Q_g*S_wet_fan_cowl

# -----------------------------------------------------------------------------------------------------------------------------------------------
# Vertical empennage

S_vert = SV #m2
laminar_vert = 0.1 
mac_vert = MACvert #m
x_c_mvert = 0.4
ttocvert = 0.12
# 0.4sweep vertical
tan_lambda_m_vert = np.tan(LEsweepvert)
AR_vert = ARvert
n_vert = x_c_mvert
m_vert = 0
taper_ratio_vert = taperingvert

def tan_lambda_n(tan_lambda_m, AR, n, m, taper_ratio):
    # Calculate the additional term in the formula
    additional_term = (4 / AR) * (n - m) * ((1 - taper_ratio) / (1 + taper_ratio))
    # Calculate tan(Λ_n)
    tan_lambda_n_value = tan_lambda_m - additional_term
    return tan_lambda_n_value


tan_lambda_n_value = tan_lambda_n(tan_lambda_m_vert, AR_vert, n_vert, m_vert, taper_ratio_vert)

lambda_mvert = np.arctan(tan_lambda_n_value)

Re = min(V_inf * mac_vert*rho/dyn_visc, 44.62*((mac_vert/k)**1.053) * M_cr**1.16)

# Laminar friction coefficient
C_f_lamvertemp = 1.328/np.sqrt(Re)

# Turbulent friction coefficient
C_f_turbvertemp =  0.455/((np.log10(Re)**(2.58))*(1+0.144*M_cr**2)**0.65)

IF_vertemp = 1.045

C_fvertemp = laminar_vert * C_f_lamvertemp + (1-laminar_vert) * C_f_turbvertemp

FF_vert_emp = (1+ (0.6 / x_c_mvert) * ttocvert + 100*ttocvert**4)*(1.34*M_cr**0.18*np.cos(lambda_mvert)**0.28) * 1.1

S_wet_emp_vert = 2*1.05*S_vert

multiplied_vertical = C_fvertemp*FF_vert_emp*IF_vertemp*S_wet_emp_vert

# ----------------------------------------------------------------------------------------------------
# horizontal empennage
S_hor = SH
laminar_hor = 0.1
mac_hor = MAChoriz
x_c_mhor = 0.4
ttochor = 0.12

tan_lambda_m_hor = np.tan(LEsweephoriz)  
AR_hor = ARhoriz
n_hor = x_c_mhor
m_hor = 0
taper_ratio_hor = taperinghoriz

tan_lambda_n_value_hor = tan_lambda_n(tan_lambda_m_hor, AR_vert, n_hor, m_hor, taper_ratio)

lambda_mhor = tan_lambda_n_value_hor

Re = min(V_inf * mac_hor*rho/dyn_visc, 44.62*((mac_hor/k)**1.053) * M_cr**1.16)

# Laminar friction coefficient
C_f_lamhoremp = 1.328/np.sqrt(Re)

# Turbulent friction coefficient
C_f_turbhoremp =  0.455/((np.log10(Re)**(2.58))*(1+0.144*M_cr**2)**0.65)

IF_horemp = 1.045

C_fhoremp = laminar_hor * C_f_lamhoremp + (1-laminar_hor) * C_f_turbhoremp

FF_hor_emp = (1+ (0.6 / x_c_mhor) * ttochor + 100*ttochor**4)*(1.34*M_cr**0.18*np.cos(lambda_mhor)**0.28) * 1.1

S_wet_emp_hor = 2*1.05*S_hor

multiplied_horizontal = C_fhoremp*FF_hor_emp*IF_horemp*S_wet_emp_hor

#-----------------------------------
# Total Cd
cd0 = (1/wing_surface)*(multiplied_engine*2 + multiplied_fus + multiplied_wing + multiplied_horizontal + multiplied_vertical) + deltaCD_wave

if __name__ == "__main__":
    print(f"The parasitic drag of the airplane is {cd0}") 
