from planform import MAC
from variables import MTOW
from fuselage import fuselage_length, l_cabin, l_nose
from variables import l_nacelle


OEW_MAC_FRAC = 0.16 #TBD #a value of 0.16 was assumed before the iterations
X_hl = 13 #assumed fixed
# group = [mass, xloc]

# wing group
wing = [0.103, 0.4 * MAC] # wing

# fuselage group
fuselage = [0.103, 0.4 * fuselage_length] # Fuselage
empennage = [0.103, 0.9 * fuselage_length] # Empennage
fixed_eq = [0.103, 0.4 * fuselage_length] # Fixed Equipment
engine = [0.103, X_hl + 0.4 * l_nacelle] # Engine + installation equipment

# Landing gear
landing_gear = [0.103] # Usage TBD


# Fuselage group CG and mass
def calculate_X_FCG(empennage, fuselage, fixedeq, engine):
    group  = [empennage, fuselage, fixedeq, engine]
    mx = 0
    m = 0
    for i in group:
        m += i[0]
        mx += i[0] * i[1]
    return mx/m , m

# Wing group CG(Relative to LEMAC) and mass
def calculate_X_WCG(wing):
    group  = [wing]
    mx = 0
    m = 0
    for i in group:
        m += i[0]
        mx += i[0] * i[1]
    return mx/m, m

# XLEMAC
def calculate_X_LEMAC(X_FCG, c_bar, x_c_WCG, M_W, M_F, x_c_OEWCG):
    # Equation as per the provided formula
    term_1 = (x_c_WCG * (M_W / M_F))
    term_2 = x_c_OEWCG * (1 + (M_W / M_F))

    X_LEMAC = X_FCG + c_bar * (term_1 - term_2)

    return X_LEMAC


#XLEMAC
X_LEMAC = calculate_X_LEMAC(calculate_X_FCG(empennage, fuselage, fixed_eq, engine)[0], MAC, calculate_X_WCG(wing)[0], calculate_X_WCG(wing)[1], calculate_X_FCG(empennage, fuselage, fixed_eq, engine)[1], OEW_MAC_FRAC)

# CGS
fuel_mass_fraction = 8100/MTOW
OEW_mass_fraction = 0.5300488240648665
payload_mass_fraction = 9302/MTOW

fuel_cg_length = X_LEMAC + (0.4 * MAC)
OEW_cg_length = X_LEMAC + OEW_MAC_FRAC * MAC
payload_cg_length = l_nose + (l_cabin/2)

#DiffCGs
OEW_WP =  ((OEW_mass_fraction * OEW_cg_length)+(payload_mass_fraction * payload_cg_length))/(OEW_mass_fraction + payload_mass_fraction)
OEW_WP_WF =  ((OEW_mass_fraction * OEW_cg_length)+(payload_mass_fraction * payload_cg_length) + (fuel_mass_fraction * fuel_cg_length))/(OEW_mass_fraction + fuel_mass_fraction + payload_mass_fraction)
OEW_WF = (((OEW_mass_fraction * OEW_cg_length)+(fuel_mass_fraction * fuel_cg_length)))/(OEW_mass_fraction + fuel_mass_fraction)

if __name__ == "__main__":
    pass
    # print(f"The different CG locations are {OEW_WP, OEW_WP_WF, OEW_WF} m")
    # print(f"x position of the main wing LEMAC is {X_LEMAC}")