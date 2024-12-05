from variables import *
from wp4_1 import Aerodynamics, InternalForces
from wp4_2 import WingBox

xflr_files = 'XFLRdata\\XFLR5sims'

# change this to any angle of attack you'd like to use
aoa = 2

# aerodynamics = Aerodynamics(folder= xflr_files, aoa= aoa, wingspan= b)
aerodynamics = Aerodynamics(folder= xflr_files, aoa= aoa, wingspan= 26.9, fus_radius=fus_radius)

g_cl, g_cd, g_cm = aerodynamics.coefficients(return_list= False)

internal_forces = InternalForces(load_factor= load_factor, sound_speed=343, half_chord_sweep= hchord_sweep, fus_radius=fus_radius, density=rho0, airspeed= airspeed, distributions= [g_cl, g_cd, g_cm], 
                                 c_r= c_r, wingspan= b, engine_z_loc= engine_z_loc, engine_length= engine_length, x_hl= x_hl, x_lemac= x_lemac, MAC= MAC, 
                                 one_engine_thrust= one_engine_thrust, fan_cowl_diameter= fan_cowl_diameter, c_t= c_r*tr)
    
    
g_shear, g_moment, g_torque, g_axial = internal_forces.internal_force_diagrams(engine_mass=engine_mass, wing_box_length=wing_box_length, 
                                        fuel_tank_length=fuel_tank_length, fuel_density=fuel_density, return_list=False)

# Plotting the internal distribution functions
internal_forces.show(engine_mass= engine_mass, wing_box_length= wing_box_length, fuel_tank_length= fuel_tank_length, fuel_density= fuel_density)


wingbox = WingBox(c_r= c_r, c_t = None, wingspan=b, area_factor_flanges=12, intersection= intersection, tr= tr, t_spar= 0.003, t_caps= 0.002)

stringers = [20, 0.9, 'L', {'base': 40e-3, 'height': 40e-3, 'thickness base': 2e-3, 'thickness height': 2e-3}]

wingbox.show(wingbox.z, load= [g_moment, g_torque], modulus= [E, G], choice= 'torsion', limit= 10, plot= True, degrees= True)