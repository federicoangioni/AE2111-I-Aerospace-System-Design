from variables import *
import matplotlib.pyplot as plt
import numpy as np

class Tension_failure():
    def __init__(self, area_func, stringers, wingspan, I_tot, M, N, chord, geometry, area_factor, t_spar, t_caps, load_factor):
        
        self.area = area_func
        
        self.geometry = geometry
        
        self.I_tot = I_tot # function of z
        
        self.halfspan = wingspan / 2
        
        self.stringers = stringers
        
        self.load_factor = load_factor
        
        self.M = M
        self.chord = chord
        
        self.area_factor = area_factor
        
        self.t_spar = t_spar
        
        self.t_caps = t_caps
        
        self.N = N
        
        
    def stress_z(self, z):

        if self.load_factor > 0: 
            a, _, _, _ = self.geometry(z)
        else:
            a, _, _, _ = self.geometry(z)

        # Evaluate the moment and inertia functions at spanwise location z
        I, _ = self.I_tot(z= z, stringers= self.stringers)

        # Calculate numerator and denominator
        numerator = (self.M(z)) * (a/2)
        denominator = I

        cross_area = self.area(chord=self.chord, geometry=self.geometry, z= z, 
                                 point_area_flange= self.area_factor, t_spar= self.t_spar, t_caps=self.t_caps, stringers= self.stringers)

        # Calculate normal stress
        sigma_z = (numerator / denominator) + abs((self.N(z)/cross_area))
        return sigma_z

    def calculate_stress_distribution(self):
        """
        Calculate the normal stress distribution along the wing span.
        """
        
        z_points = self.halfspan
        
        stress_distribution = []
        for z in z_points:
            y = y_func(z)
            sigma_z = stress_z(Mx_func, Ax_func, Ixx_func, Area_func, z, y)
            stress_distribution.append(sigma_z * 1.5)

        return np.array(stress_distribution)

def main():
    # Define spanwise locations
    z_points = np.linspace(0, b/2, 100)  # Wing span from 0 to 10 meters

    # Calculate stress distribution
    stress_distribution = calculate_stress_distribution(
        z_points=z_points,
        y_func=y_func,
        Mx_func=Mx_func,
        Ax_func=Ax_func,
        Ixx_func=Ixx_func,
        Area_func=Area_func
    )

    # Find critical stress
    # max_stress = max(stress_distribution)
    # critical_z = z_points[np.argmax(stress_distribution)]
    # print(f"Maximum Normal Stress: {max_stress:.2f} Pa at Spanwise Location: {critical_z:.2f} m")

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

    def margin_of_safety(): # z = location along span
        margin_of_safety = 440000000/stress_distribution
        return margin_of_safety # should be greater than 1 everywhere, otherwise expect failure, can come close to 1 tho not lower

    MOS_tens_tab = []
    z_tab = []

    def graph_tension_MOS():
        z_values = np.linspace(1, 26.9/2, 1000)
        i=0
        print(t_caps, t_spar, area_factor)
        
        for z in z_points:
            MOS_tens_tab.append(margin_of_safety()[i])
            z_tab.append(z)
            i+=1
        plt.plot(z_points, MOS_tens_tab, label="Normal Stress Distribution", color="blue")
        plt.axhline(y=1, color='r')
        plt.title('Margin of safety factor visualization along wing')
        plt.xlabel('Spanwise location')
        plt.ylabel('Margin of safety')
        plt.ylim(0, 8)  # Set the y-axis limits
        plt.show()


    graph_tension_MOS() 


main()

# from wp5_1 import Area_crosssection
# from wp4_2 import WingBox
# from variables import *
# import matplotlib.pyplot as plt

# # t_caps = 0.008
# # t_spar = 0.008

# # area_factor = 12

# wingbox = WingBox(c_r= c_r, c_t = None, wingspan=b, area_factor_flanges=area_factor, intersection= intersection, tr= tr, t_spar= t_spar, t_caps= t_caps)

# #stringers = [24, 1, 'L', {'base': 30e-3, 'height': 30e-3, 'thickness base': 2e-3, 'thickness height': 2e-3}]

# def x_func(z):
#     return 0 * z

# def y_func(z):

#     if LOADFACTOR > 0: 
#         y_loc = wingbox.geometry(2*z/b)[0] / 2
#     else:
#         y_loc = -wingbox.geometry(2*z/b)[0] / 2
#     return y_loc

# def Ixx_func(z):
#     return wingbox.MOM_total(2*z/b, stringers=stringers)[0]

# def Iyy_func(z):
#     return wingbox.MOM_total(2*z/b, stringers=stringers)[1]

# def Ixy_func(z):
#     return 0 * 2* z/b

# def Mx_func(z):
#     return g_moment(z)

# def My_func(z):
#     return 0 * z

# def Ax_func(z):
#     return g_axial(z)

# def Area_func(z):
#     return Area_crosssection(wingbox.chord, wingbox.geometry, z, area_factor, t_spar, t_caps, stringers)

# print(wingbox.MOM_total(z=0, stringers=stringers)[0])

# def stress_z(Mx_func, My_func, Ax_func, Ixx_func, Iyy_func, Ixy_func, Area_func, z, x, y):

#     # Evaluate the moment and inertia functions at spanwise location z
#     Mx = Mx_func(z)
#     My = My_func(z)
#     Ixx = Ixx_func(z)
#     Iyy = Iyy_func(z)
#     Ixy = Ixy_func(z)
#     Ax = Ax_func(z)
#     A_func = Area_func(z)

#     # Calculate numerator and denominator
#     numerator = (Mx * Iyy - My * Ixy) * y + (My * Ixx - Mx * Ixy) * x
#     denominator = Ixx * Iyy - Ixy**2

#     # division by zero?
#     if denominator == 0:
#         raise ValueError("Denominator is zero. Check inertia values.")

#     # Calculate normal stress
#     sigma_z = (numerator / denominator) + (Ax/A_func)
#     return sigma_z

# def calculate_stress_distribution(z_points, x_func, y_func, Mx_func, My_func, Ax_func, Ixx_func, Iyy_func, Ixy_func, Area_func):
#     """
#     Calculate the normal stress distribution along the wing span.
#     """
#     stress_distribution = []
#     for z in z_points:
#         x = x_func(z)
#         y = y_func(z)
#         sigma_z = stress_z(Mx_func, My_func, Ax_func, Ixx_func, Iyy_func, Ixy_func, Area_func, z, x, y)
#         stress_distribution.append(sigma_z * 1.5)

#         # print(y)
#         # print(Mx_func(z))
#         #print(Ixx_func(z))

#     return np.array(stress_distribution)

# def main():
#     # Define spanwise locations
#     z_points = np.linspace(0, b/2, 100)  # Wing span from 0 to 10 meters

#     # Calculate stress distribution
#     stress_distribution = calculate_stress_distribution(
#         z_points=z_points,
#         x_func=x_func,
#         y_func=y_func,
#         Mx_func=Mx_func,
#         My_func= My_func,  # Assuming same moment for simplicity
#         Ax_func=Ax_func,
#         Ixx_func=Ixx_func,
#         Iyy_func=Iyy_func,
#         Ixy_func=Ixy_func,
#         Area_func=Area_func
#     )

#     # Find critical stress
#     # max_stress = max(stress_distribution)
#     # critical_z = z_points[np.argmax(stress_distribution)]
#     # print(f"Maximum Normal Stress: {max_stress:.2f} Pa at Spanwise Location: {critical_z:.2f} m")

#     # Plot stress distribution
#     plt.figure(figsize=(12, 6))
#     plt.plot(z_points, stress_distribution/ 1e6, label="Normal Stress Distribution", color="blue")
#     plt.axhline(0, color="black", linewidth=0.8)
#     plt.xlabel("Spanwise Location (z) [m]", fontsize=12)
#     plt.ylabel("Normal Stress [MPa]", fontsize=12)
#     plt.title("Normal Stress Distribution Along the Wing Span", fontsize=14)
#     plt.grid(True)
#     plt.legend()
#     plt.show()

#     def margin_of_safety(): # z = location along span
#         margin_of_safety = 440000000/stress_distribution
#         return margin_of_safety # should be greater than 1 everywhere, otherwise expect failure, can come close to 1 tho not lower

#     MOS_tens_tab = []
#     z_tab = []

#     def graph_tension_MOS():
#         z_values = np.linspace(1, 26.9/2, 1000)
#         i=0
#         print(t_caps, t_spar, area_factor)
        
#         for z in z_points:
#             MOS_tens_tab.append(margin_of_safety()[i])
#             z_tab.append(z)
#             i+=1
#         plt.plot(z_points, MOS_tens_tab, label="Normal Stress Distribution", color="blue")
#         plt.axhline(y=1, color='r')
#         plt.title('Margin of safety factor visualization along wing')
#         plt.xlabel('Spanwise location')
#         plt.ylabel('Margin of safety')
#         plt.ylim(0, 8)  # Set the y-axis limits
#         plt.show()

#         print(Mx_func(0))

#     graph_tension_MOS() 

# if __name__ == "__main__":
#     main()