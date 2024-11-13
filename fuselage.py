import numpy as np

# Fuselage Variables
pass1_mass = 76
pass1_volume = 0.7
carryon1_mass = 7
mass1_lugg = 10

passengers = 68

mass_pass = passengers*pass1_mass
mass_lugg = passengers*carryon1_mass
mass_cargolugg = passengers * mass1_lugg

rho_cargo = 250 # kg/m3
rho_density = 180
passengers = 68

n_sa  =  np.ceil(0.45*np.sqrt(passengers))

w_aisle_armrest = 0.52
w_aisle_shoulder = 0.62
cleareance = 0.05
n_aisles = 1
w_seat = 0.42
w_armrest = 0.05
# ----------------------------------------------------------------------------------------------


print(design_payload *1e3/(mass1_lugg+carryon1_mass+pass1_mass))
print(f"Passengers mass only {mass_pass} kg with {passengers} passengers")
print(f"Design payload mass is: {design_payload*10**3} kg")
print(f"Remaining cargo mass left for payload is: {design_payload*1e3 - (mass_pass + mass_lugg + mass_cargolugg)}")

w_cabin = n_sa*w_seat + (n_sa+ n_aisles + 1)*w_armrest + n_aisles*w_aisle_armrest + cleareance*2
w_floor = w_cabin - 2*(w_armrest+cleareance)
w_headroom = w_floor - w_seat