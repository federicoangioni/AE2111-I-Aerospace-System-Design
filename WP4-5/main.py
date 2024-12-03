from variables import *
from wp4_1 import Aerodynamics, InternalForces
from wp4_2 import WingBox

xflr_files = 'XFLRdata/XFLR5sims'

load_factor = 2.5
hchord_sweep = 22.4645
fus_radius = d/2
engine_z_loc = 4.35 # m

# change this to any angle of attack you'd like to use
aoa = 5

# aerodynamics = Aerodynamics(folder= xflr_files, aoa= aoa, wingspan= b)
aerodynamics = Aerodynamics(folder= xflr_files, aoa= aoa, wingspan= 26.9, fus_radius=fus_radius)

g_cl, g_cd, g_cm = aerodynamics.coefficients(return_list= False)



internal_forces = InternalForces(load_factor= load_factor, half_chord_sweep= hchord_sweep, fus_radius=fus_radius, density=rho0, airspeed= airspeed, distributions= [g_cl, g_cd, g_cm], 
                                 c_r= c_r, wingspan= b, engine_z_loc= engine_z_loc, engine_length= engine_length, x_hl, x_lemac, MAC, one_engine_thrust, fan_cowl_diameter, c_t)
    
    
    
    #c_r= c_r, wingspan= b, density= rho0, airspeed= 66.26, 
    #                            distributions= [g_cl, g_cd, g_cm], engine_z_loc= 4.351,tr= tr)


internal_forces = InternalForces(c_r= 4.33, wingspan= 26.9, density= 1.225, airspeed= 66.26, 
                                 distributions= [g_cl, g_cd, g_cm], engine_z_loc= 4.351,tr= 0.31)


# plotting of internal forces
internal_forces.show(engine_mass=2306, wingbox_height= 0.6, fuel_tank_length= 13.45, fuel_density=800)

