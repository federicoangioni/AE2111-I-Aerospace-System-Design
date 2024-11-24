from variables import *
from wp4_1 import Aerodynamics, InternalForces
from wp4_2 import WingBox

xflr_files = 'XFLRdata/XFLR5sims'

# change this to any angle of attack you'd like to use
aoa = 5

aerodynamics = Aerodynamics(folder= xflr_files, aoa= aoa, wingspan= b)

g_cl, g_cd, g_cm = aerodynamics.coefficients(return_list= False)

# plotting of aerodynamic curves
aerodynamics.show()

internal_forces = InternalForces(c_r= c_r, wingspan= b, density= rho0, airspeed= 228, 
                                 distributions= [g_cl, g_cd, g_cm], tr= tr)

# plotting of internal forces
# internal_forces.show()