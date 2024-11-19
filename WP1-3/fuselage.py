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

# ----------------------------------------------------------------------------------------------
# cross-section 
w_aisle_armrest = 0.52
w_aisle_shoulder = 0.62
cleareance = 0.05
n_aisles = 1
w_seat = 0.42
w_armrest = 0.05
design_payload = 6.355 

w_cabin = n_sa*w_seat + (n_sa+ n_aisles + 1)*w_armrest + n_aisles*w_aisle_armrest + cleareance*2
w_floor = w_cabin - 2*(w_armrest+cleareance)
w_headroom = w_floor - w_seat

d_fus = 2.736

# ----------------------------------------------------------------------------------------------
# top view

seat_pitch = 0.76 #m

k_cabin = 1.08 # this value can be changed due to statistical analysis
l_cabin = k_cabin*passengers/n_sa

l_nose = 4. # nose length in m
nose_cone_sl = 2. # nose cone slenderness ratio
tail_cone_sl = 3.5 # tail cone slenderness ratio
tttc_lr = .75 # tail to tail-cone length ratio
l_nc = nose_cone_sl*d_fus
l_t = tail_cone_sl*tttc_lr*d_fus
l_tc = tail_cone_sl *d_fus
fuselage_length = l_cabin + l_nose + tail_cone_sl*tttc_lr*d_fus


if __name__ == "__main__":
    pass
    # print(design_payload *1e3/(mass1_lugg+carryon1_mass+pass1_mass))
    # print(f"Passengers mass only {mass_pass} kg with {passengers} passengers")
    # print(f"Design payload mass is: {design_payload*10**3} kg")
    # print(f"Remaining cargo mass left for payload is: {design_payload*1e3 - (mass_pass + mass_lugg + mass_cargolugg)}")
    # print(f"The fuselage length is {fuselage_length} m")
    # print(f"The tail length is (l_t) {l_t} m")
    # print(f"The nose cone length is (l_nc) {l_nc} m")
    # print(f"The length of the cabin l_cab is {l_cabin} m")
    # print(f"Length of the tail cone is l_tc {l_tc} m")
    # print(f"length of the nose is l_n {l_nose} m")