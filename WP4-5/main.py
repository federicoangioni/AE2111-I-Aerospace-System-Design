from variables import *
from wp4_1 import Aerodynamics, InternalForces
from wp4_2 import WingBox
import numpy as np

xflr_files = 'XFLRdata\\XFLR5sims'

load_factor = 2.5
hchord_sweep = 22.4645 # m
fus_radius = d/2 # m
engine_z_loc = 4.35 # m

# change this to any angle of attack you'd like to use
aoa = 5
z = np.linspace(0, 1, 1000)


# aerodynamics = Aerodynamics(folder= xflr_files, aoa= aoa, wingspan= b)
aerodynamics = Aerodynamics(folder= xflr_files, aoa= aoa, wingspan= 26.9, fus_radius=fus_radius)

g_cl, g_cd, g_cm = aerodynamics.coefficients(return_list= False)



internal_forces = InternalForces(load_factor= load_factor, half_chord_sweep= hchord_sweep, fus_radius=fus_radius, density=rho0, airspeed= airspeed, distributions= [g_cl, g_cd, g_cm], 
                                 c_r= c_r, wingspan= b, engine_z_loc= engine_z_loc, engine_length= engine_length, x_hl= x_hl, x_lemac= x_lemac, MAC= MAC, 
                                 one_engine_thrust= one_engine_thrust, fan_cowl_diameter= fan_cowl_diameter, c_t= c_r*tr)
    
    
g_shear, g_moment, g_torque, g_axial = internal_forces.internal_force_diagrams(engine_mass=engine_mass, wing_box_length=wing_box_length, 
                                        fuel_tank_length=fuel_tank_length, fuel_density=fuel_density, return_list=False)

# Plotting the internal distribution functions
# internal_forces.show(engine_mass= engine_mass, wing_box_length= wing_box_length, fuel_tank_length= fuel_tank_length, fuel_density= fuel_density)
t = 0.005 # m

wingbox = WingBox(t= t, c_r= c_r, c_t = None, wingspan=b, intersection=fus_radius, tr= tr)

wingbox.torsion(z=z, )


