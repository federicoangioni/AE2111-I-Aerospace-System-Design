from variables import *
from wp4_1 import Aerodynamics, InternalForces
from wp4_2 import WingBox
from wp4_3 import VelocityLoadFactorDiagram, LoadCases

xflr_files = 'XFLRdata/XFLR5sims'

# change this to any angle of attack you'd like to use
aoa = 5

# aerodynamics = Aerodynamics(folder= xflr_files, aoa= aoa, wingspan= b)
aerodynamics = Aerodynamics(folder= xflr_files, aoa= aoa, wingspan= 26.9)

g_cl, g_cd, g_cm = aerodynamics.coefficients(return_list= False)

# plotting of aerodynamic curves
# aerodynamics.show()

# internal_forces = InternalForces(c_r= c_r, wingspan= b, density= rho0, airspeed= 66.26, 
#                                 distributions= [g_cl, g_cd, g_cm], engine_z_loc= 4.351,tr= tr)


internal_forces = InternalForces(c_r= 4.33, wingspan= 26.9, density= 1.225, airspeed= 66.26, 
                                 distributions= [g_cl, g_cd, g_cm], engine_z_loc= 4.351,tr= 0.31)


# plotting of internal forces
internal_forces.show(engine_mass=2306, wingbox_height= 0.6, fuel_tank_length= 13.45, fuel_density=800)

