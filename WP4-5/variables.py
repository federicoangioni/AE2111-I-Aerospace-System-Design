c_r = 4.33 #root cord
tr = 0.31 #taper ratio
b = 26.9 #wingspan
d = 2.74 #fuselage diameter
alpha =1.48414 #in degrees
airspeed = 228.6

# Engine specs
engine_length = 3.0 # m
x_hl = 13
x_lemac = 16.37
MAC=3.05
one_engine_thrust= 78466 # in Newtons
fan_cowl_diameter=1.448

engine_mass= 2306        # Mass of the engine in kilograms
wing_box_length= 0.55     # Length of the wingbox relative to chord
fuel_tank_length= 11.98   # Length of the fuel tank in meters
fuel_density= 800        # Density of the fuel in kg/m^3

#material properties Aluminum 2024-T81
E = 72_400_000_000 #Young modulus [Pa]
G = 28_000_000_000 #shear modulus [Pa]
rho = 2780 #density [kg/m^3]
tsY = 450000000 #tensile strength yield [Pa]
tsU = 485000000 #tensile strength ultimate [Pa]

# Atmospheric properties
rho0 = 0.409727 #kg/m^3
T0 = 222.770 #K
P0 = 26200.8 #Pa
gamma = 1.4
R = 287.05 #J/(kgK)
g = 9.81 #m/s^2

load_factor = -1
hchord_sweep = 22.4645 # m
fus_radius = d/2 # m
engine_z_loc = 4.35 # m
intersection = (fus_radius)/(b/2)