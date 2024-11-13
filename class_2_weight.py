import numpy as np
from variables import design_weight, limit_load_factor, MTOW
from SARoptimization import optimized_S, optimized_AR, tr
from planform import angle_at_xdivc, MAC, c4sweep, b
from CG_location import X_LEMAC
from empennage_planform import SH, bhoriz, MAChoriz, LEsweephoriz, ARhoriz, taperinghoriz, bvert, SV, MACvert, LEsweepvert, ARvert, taperingvert, bmacvert, bmachoriz, LERoot_horiz, LERoot_vert
from fuselage import fuselage_length, l_nc, l_cabin, l_tc, d_fus
### Variables names and value
# General Empennage
N_z = 1.5 * limit_load_factor 
wing_macx = X_LEMAC + 0.25 * MAC #m 

#---------------------------------------------------------------------------------------------------

# Horizontal empennage
K_uht = 1.0
F_w = 2.2 # m # fuselage width at horizontal tail intersection
S_ht = SH #m2
mac_h = MAChoriz # m 
c0sweep_ht = LEsweephoriz # horizontal sweep at mac DEGREES 
L_t_h = bmachoriz*np.tan(c0sweep_ht)+0.25*mac_h + LERoot_horiz - wing_macx #wing quarter mac to tail quarter mac
K_y = 0.3*L_t_h
A_h = ARhoriz # AR horizontal tail
S_e = 1 # m2 # elevator area
tr_h = taperinghoriz
b_h = bhoriz # m 
c4sweep_ht = angle_at_xdivc(1, 4, LEsweep = c0sweep_ht, c_r = 3.62, tr = tr_h, b = b_h) # in radians

#---------------------------------------------------------------------------------------------------
# Vertical tail
Ht = 0.0 # ok
Hv = bvert # m 
S_vt = SV  # m2
mac_v = MACvert #m 
c0sweep_vt = LEsweepvert # sweep at 0 
L_t_v =  bmacvert*np.tan(c0sweep_vt)+0.25*mac_v + LERoot_vert - wing_macx #m #wing quarter mac to tail quarter mac

K_z = L_t_v # m radius of gyration
A_v = ARvert # AR vertical tail
tc_root = 0.12
tr_v = taperingvert
b_v = Hv # m 
c4sweep_vt = angle_at_xdivc(1, 4, LEsweep = c0sweep_vt, c_r = 3.62, tr = tr_v, b = b_v) # in radians
#---------------------------------------------------------------------------------------------------
# Main wing
S_w = optimized_S[0]
t_to_c = 0.14

S_csw = (1.2387 * 2 + 22.56) # FIX THIS PLEASEEEEEN XXXXXXXXXXXXXXXX 
AR = optimized_AR

#---------------------------------------------------------------------------------------------------
# fuselage weight

K_door = 1.12
k_LG = 1.12
L1_fus = l_nc * 3.28084
L2_fus = l_cabin* 3.28084
L3_fus = l_tc * 3.28084

S_f = (np.pi * (d_fus* 3.28084) / 4) * ((1 / (3 * L1_fus**2)) * ((4 * L1_fus**2 + (d_fus* 3.28084)**2 / 4)**1.5 - (d_fus* 3.28084)**3 / 8)) - (d_fus* 3.28084) + 4 * L2_fus + 2 * np.sqrt(L3_fus**2 + (d_fus* 3.28084)**2 / 4)

K_ws = 0.75*((1+2*tr)/(1+tr)) * (b*3.28084*np.tan(c4sweep/(fuselage_length*3.28084)))

def wing_weight(Wdg, Nz, Sw, A, tc_root, lamda, Scsw):
    # Calculate wing weight using the given formula
    W_wing = 0.0051 * ((Wdg * Nz) ** (0.557)) * (Sw ** 0.649) * (A ** 0.5) * ((tc_root) ** (-0.4)) * ((1 + lamda) ** 0.1) * (np.cos(lamda) ** (-1.0)) * (Scsw ** 0.1)
    return W_wing

def fuselage_weight(K_door, K_Lg, Wdg, Nz, L, Sf, K_ws, D):
    """
    K_door 1.12
    K_Lg 1.12
    Wdg design_weight
    Nz
    L fuselage length
    Sf fuselage wetted area
    K_ws 0.75*((1+2*tr)/(1+tr)) * (B_w*np.tan(Lambda_c4/L_fus))
    D fuselage diameter
    """
    W_fuselage = 0.3280 * K_door * K_Lg * ((Wdg * Nz) ** 0.5) * (L ** 0.25) * (Sf ** 0.302) * ((1 + K_ws) ** 0.04) * ((L/D) ** 0.1)
    return W_fuselage
  
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

fus_weight = fuselage_weight(K_door=K_door, K_Lg=k_LG, Wdg = design_weight* 2.20462, Nz=N_z, L=fuselage_length* 3.28084, Sf=S_f, K_ws=K_ws, D= d_fus*3.28084)

wingweight =  wing_weight(Wdg = design_weight * 2.20462, Nz = N_z, Sw = S_w * 10.7639, A = AR, tc_root=t_to_c, lamda =c4sweep , Scsw = S_csw * 10.7639)

v_weight_GD = vertical_weight_GD(z_h=0, b_v = b_v* 3.28084, W_to = MTOW *2.20462, n_ult = N_z, S_v = S_vt*10.7639, M_H = 0.77, l_v = L_t_h * 3.28084, S_r = 1, A_v = A_v, tr_v = tr_v, c4_vertical=c4sweep_vt)
  
h_weight = horizontal_tail_weight(K_uht = K_uht, F_w = F_w * 3.28084, B_h = b_h * 3.28084, Wdg = design_weight * 2.20462, Nz = N_z, S_ht = S_ht*10.7639, L_t = L_t_h * 3.28084, K_y = K_y * 3.28084, Lambda_ht = c4sweep_ht, A_h = A_h, S_e = S_e * 10.7639)

v_weight = vertical_tail_weight(Ht*3.28084, Hv*3.28084, Wdg = design_weight * 2.20462, Nz = N_z, L_t = L_t_v * 3.28084, S_vt = S_vt*10.7639, K_z = K_z**3.28084, Lambda_vt = c4sweep_vt, A_v = A_v, tc_root = tc_root)

if __name__ == "__main__":
  print(K_door, k_LG, design_weight, N_z, fuselage_length, S_f, K_ws, d_fus, c4sweep)
  print(f"weight fuselage {fus_weight/2.20462}")
  print(f"weight wing {wingweight/2.20462} kg (Raymer)")
  print(f"Vertical tail weight is {v_weight/2.20462} kg (Raymer) DO NOT USE")
  print(f"Horizontal tail weight is {h_weight/2.20462} kg (Raymer)")
  print(f"Vertical tail weight from GD estimation {v_weight_GD/2.20462} kg (GD method)")