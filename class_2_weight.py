import numpy as np
from variables import MTOW, landing_mass_fraction, limit_load_factor
from SARoptimization import optimized_S, optimized_AR, tr


mtow_kg = MTOW # kg
mlw_kg = mtow_kg*landing_mass_fraction

design_weight = ((mtow_kg + mlw_kg)/2) # kg

# formula for different sweep angles
def angle_at_xdivc(x, c, LEsweep, c_r, tr, b):
    return np.arctan(np.tan(LEsweep) - (x/c)  * 2 * (c_r/b) * (1-tr))

### Variables names and value

# General Empennage
N_z = 1.5 * limit_load_factor 
wing_macx = 16.63 + 0.25 *3.06 #m 

#---------------------------------------------------------------------------------------------------

# Horizontal empennage
K_uht = 1.0
F_w = 2.2 # m # fuselage width at horizontal tail intersection
B_h = 10.13 #m
S_ht = 25.66 #m2
mac_h = 2.69 # m 
c0sweep_ht = np.radians(34.4) # horizontal sweep at mac DEGREES 
L_t_h = 1.09*np.tan(c0sweep_ht)+0.25*mac_h + 24.9 - wing_macx #wing quarter mac to tail quarter mac
K_y = 0.3*L_t_h
A_h = 4 # AR horizontal tail
S_e = 1 # m2 # elevator area
tr_h = 0.4
b_h = 10.13 # m 
c4sweep_ht = angle_at_xdivc(1, 4, LEsweep = c0sweep_ht, c_r = 3.62, tr = tr_h, b = b_h) # in radians

#---------------------------------------------------------------------------------------------------
# Vertical tail
Ht = 0.0 # ok
Hv = 5.00 # m 
S_vt = 14.74  # m2
mac_v = 3.15 #m 
c0sweep_vt = np.radians(48.42) # sweep at 0 
L_t_v =  2.11*np.tan(c0sweep_vt)+0.25*mac_v + 23.1 - wing_macx #m #wing quarter mac to tail quarter mac
K_z = L_t_v # m radius of gyration
A_v = 1.75 # AR vertical tail
tc_root = 0.12
tr_v = 0.33
b_v = 5 # m 
c4sweep_vt = angle_at_xdivc(1, 4, LEsweep = c0sweep_vt, c_r = 3.62, tr = tr_v, b = b_v) # in radians
#---------------------------------------------------------------------------------------------------
# Main wing
S_w = optimized_S[0]
t_to_c = 0.14
Lambda_c4 = np.radians(25)
S_csw = (1.2387 * 2 + 22.56)
AR = optimized_AR

#---------------------------------------------------------------------------------------------------

def wing_weight(Wdg, Nz, Sw, A, tc_root, lamda, Scsw):
    # Calculate wing weight using the given formula
    W_wing = 0.0051 * ((Wdg * Nz) ** (0.557)) * (Sw ** 0.649) * (A ** 0.5) * ((tc_root) ** (-0.4)) * ((1 + lamda) ** 0.1) * (np.cos(lamda) ** (-1.0)) * (Scsw ** 0.1)
    return W_wing


def horizontal_tail_weight(K_uht, F_w, B_h, Wdg, Nz, S_ht, L_t, K_y, Lambda_ht, A_h, S_e):
    W_ht = 0.0379 * K_uht * ((1 + F_w / B_h) ** -0.25) * (Wdg ** 0.639) * (Nz ** 0.10) * (S_ht ** 0.75) * (L_t ** -1.0) * (K_y ** 0.704) * ((np.cos(Lambda_ht)) ** -1.0) * (A_h ** 0.166) * ((1 + S_e / S_ht) ** 0.1)
    return W_ht

def vertical_tail_weight(Ht, Hv, Wdg, Nz, L_t, S_vt, K_z, Lambda_vt, A_v, tc_root):
  W_vertical_tail = (0.0026 * (1 + Ht / Hv)**0.225 * Wdg**0.556 * Nz**0.536 * L_t**-0.5 * S_vt**0.5 * K_z**0.875 *(np.cos((Lambda_vt)))**-1 * A_v**0.35 * (tc_root)**-0.5)
  return W_vertical_tail

def vertical_weight_GD(z_h, b_v, W_to, n_ult, S_v, M_H, l_v, S_r, A_v, tr_v, c4_vertical):
  
  """
  z_h 0 for fuselage mounted horizontal tails
  l_v dist. from wing c/4 to vert tail c_v/4 in ft
  tr_v taper ratio ft2
  S_r rudder surface area ft2
  S_v vertical tail surface area ft2
  """
  W_v = 0.19*((1+z_h/b_v)**0.5 * (W_to*n_ult)**0.363 * S_v **1.089 * M_H **0.601 *l_v**-0.726 * (1+S_r/S_v)**0.217 * A_v **0.337 *(1+tr_v)**0.363 * np.cos(c4_vertical)**-0.484)**1.014
  return W_v

wingweight =  wing_weight(Wdg = design_weight * 2.20462, Nz = N_z, Sw = S_w * 10.7639, A = AR, tc_root=t_to_c, lamda =Lambda_c4 , Scsw = S_csw * 10.7639)

v_weight_GD = vertical_weight_GD(z_h=0, b_v = b_v* 3.28084, W_to = MTOW *2.20462, n_ult = N_z, S_v = S_vt*10.7639, M_H = 0.77, l_v = L_t_h * 3.28084, S_r = 1, A_v = A_v, tr_v = tr_v, c4_vertical=c4sweep_vt)
  
h_weight = horizontal_tail_weight(K_uht = K_uht, F_w = F_w * 3.28084, B_h = b_h * 3.28084, Wdg = design_weight * 2.20462, Nz = N_z, S_ht = S_ht*10.7639, L_t = L_t_h * 3.28084, K_y = K_y * 3.28084, Lambda_ht = c4sweep_ht, A_h = A_h, S_e = S_e * 10.7639)

v_weight = vertical_tail_weight(Ht*3.28084, Hv*3.28084, Wdg = design_weight * 2.20462, Nz = N_z, L_t = L_t_v * 3.28084, S_vt = S_vt*10.7639, K_z = K_z**3.28084, Lambda_vt = c4sweep_vt, A_v = A_v, tc_root = tc_root)

if __name__ == "__main__":
  print(f"weight wing {wingweight/2.20462} kg (Raymer)")
  print(f"Vertical tail weight is {v_weight/2.20462} kg (Raymer)")
  print(f"Horizontal tail weight is {h_weight/2.20462} kg (Raymer)")
  print(f"Vertical tail weight from GD estimation {v_weight_GD/2.20462} kg (GD method)")