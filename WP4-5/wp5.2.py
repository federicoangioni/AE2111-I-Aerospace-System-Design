import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
#from main import g_moment, wingbox
#from variables import b
from variables import *
from wp4_1 import Aerodynamics, InternalForces
from wp4_2 import WingBox

xflr_files = 'XFLRdata\\XFLR5sims'

# change these
aircraft_mass = 35688
alt_sound_speed = 296.56

internal_forces = InternalForces(aircraft_mass=aircraft_mass, load_factor= 2.5, sound_speed=alt_sound_speed, half_chord_sweep= hchord_sweep, fus_radius=fus_radius, density=rho0, airspeed= airspeed, 
                                 c_r= c_r, wingspan= b, engine_z_loc= engine_z_loc, engine_length= engine_length, x_hl= x_hl, x_lemac= x_lemac, MAC= MAC, 
                                 one_engine_thrust= one_engine_thrust, fan_cowl_diameter= fan_cowl_diameter, c_t= c_r*tr)
    
    
g_shear, g_moment, g_torque, g_axial = internal_forces.force_diagrams(engine_mass=engine_mass, wing_box_length=wing_box_length, 
                                        fuel_tank_length=fuel_tank_length, fuel_density=fuel_density)[4:]


wingbox = WingBox(c_r= c_r, c_t = None, wingspan=b, area_factor_flanges=12, intersection= intersection, tr= tr, t_spar= 0.004, t_caps= 0.004)

stringers = [20, 1, 'L', {'base': 30e-3, 'height': 30e-3, 'thickness base': 2e-3, 'thickness height': 2e-3}]


print(stringers)


def x_func(z):
    return 0 

def y_func(z):
    return wingbox.geometry(2*z/b)[0] / 2

def Ixx_func(z):
    return wingbox.MOM_total(2*z/b, stringers=stringers)[0]

def Iyy_func(z):
    return wingbox.MOM_total(2*z/b, stringers=stringers)[1]

def Ixy_func(z):
    return 0 * 2* z/b

def Mx_func(z):
    return g_moment(z)

def My_func(z):
    return 0 * z

def Ax_func(z):
    return g_axial(z)

#def Area_func(z):
#   return 

print(wingbox.MOM_total(z=0, stringers=stringers)[0])


def stress_z(Mx_func, My_func, Ax_func, Ixx_func, Iyy_func, Ixy_func, z, x, y):

    # Evaluate the moment and inertia functions at spanwise location z
    Mx = Mx_func(z)
    My = My_func(z)
    Ixx = Ixx_func(z)
    Iyy = Iyy_func(z)
    Ixy = Ixy_func(z)
    Ax = Ax_func(z)

    # Calculate numerator and denominator
    numerator = (Mx * Iyy - My * Ixy) * y + (My * Ixx - Mx * Ixy) * x
    denominator = Ixx * Iyy - Ixy**2

    # division by zero?
    if denominator == 0:
        raise ValueError("Denominator is zero. Check inertia values.")

    # Calculate normal stress
    sigma_z = (numerator / denominator) + # (Ax/)
    return sigma_z

def calculate_stress_distribution(z_points, x_func, y_func, Mx_func, My_func, Ixx_func, Iyy_func, Ixy_func):
    """
    Calculate the normal stress distribution along the wing span.
    """
    stress_distribution = []
    for z in z_points:
        x = x_func(z)
        y = y_func(z)
        sigma_z = stress_z(Mx_func, My_func, Ixx_func, Iyy_func, Ixy_func, z, x, y)
        stress_distribution.append(sigma_z)



        # print(y)
        # print(Mx_func(z))
        #print(Ixx_func(z))


    return np.array(stress_distribution)

def main():
    # Define spanwise locations
    z_points = np.linspace(0, b/2, 100)  # Wing span from 0 to 10 meters

    # Calculate stress distribution
    stress_distribution = calculate_stress_distribution(
        z_points=z_points,
        x_func=x_func,
        y_func=y_func,
        Mx_func=Mx_func,
        My_func= My_func,  # Assuming same moment for simplicity
        Ixx_func=Ixx_func,
        Iyy_func=Iyy_func,
        Ixy_func=Ixy_func
    )

    # Find critical stress
    max_stress = max(stress_distribution)
    critical_z = z_points[np.argmax(stress_distribution)]
    print(f"Maximum Normal Stress: {max_stress:.2f} Pa at Spanwise Location: {critical_z:.2f} m")

    # Plot stress distribution
    plt.figure(figsize=(12, 6))
    plt.plot(z_points, stress_distribution/ 1e6, label="Normal Stress Distribution", color="blue")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Spanwise Location (z) [m]", fontsize=12)
    plt.ylabel("Normal Stress [MPa]", fontsize=12)
    plt.title("Normal Stress Distribution Along the Wing Span", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()


