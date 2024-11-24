import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sympy as sp
import os

# authors: Medhansh, Teodor
 
class Aerodynamics():
    def __init__(self,  folder: str, aoa: int, wingspan: int):
        self.files = [os.path.join(folder, file) for file in os.listdir(folder)] # makes a list with all the files in the XFLR folder
        self.aoa = aoa
        self.wingspan = wingspan
        
    def coefficients(self, return_list: bool):
        ylst = np.genfromtxt(self.files[self.aoa], skip_header=40, max_rows=19, usecols=(0,), invalid_raise=False)
        Cllst = np.genfromtxt(self.files[self.aoa], skip_header=40, max_rows=19, usecols=(3,), invalid_raise=False)
        Cdlst = np.genfromtxt(self.files[self.aoa], skip_header=40, max_rows=19, usecols=(5,), invalid_raise=False)
        Cmlst = np.genfromtxt(self.files[self.aoa], skip_header=40, max_rows=19, usecols=(7,), invalid_raise=False)

        # Interpolate data
        g_cl = interp1d(ylst, Cllst, kind='cubic', fill_value="extrapolate") # Cl scipy function, callable
        g_cd = interp1d(ylst, Cdlst, kind='cubic', fill_value="extrapolate") # Cd scipy function, callable
        g_cm = interp1d(ylst, Cmlst, kind='cubic', fill_value="extrapolate") # Cm scipy functio, callable
        
        if return_list:    
            return g_cl, g_cd, g_cm, ylst, Cllst, Cdlst, Cmlst
        else:
            return g_cl, g_cd, g_cm
    
    def show(self):
        z_points = np.linspace(0, self.wingspan / 2)
        self.g_cl, self.g_cd, self.g_cm, self.ylst, self.Cllst, self.Cdlst, self.Cmlst = self.coefficients(True)
        
        Cl_points = self.g_cl(z_points)
        Cd_points = self.g_cd(z_points)
        Cm_points = self.g_cm(z_points)
        
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Plot Cl
        axs[0].plot(self.ylst, self.Cllst, 'o', label='Original $C_L$', markersize=8)
        axs[0].plot(z_points, Cl_points, '-', label='Cubic Interpolation')
        axs[0].set_xlabel('y')
        axs[0].set_ylabel('$C_L$')
        axs[0].set_title('Cubic Interpolation of $C_L$ vs y')
        axs[0].legend()
        axs[0].grid(True)

        # Plot Cd
        axs[1].plot(self.ylst, self.Cdlst, 'o', label='Original $C_D$', markersize=8)
        axs[1].plot(z_points, Cd_points, '-', label='Cubic Interpolation')
        axs[1].set_xlabel('y')
        axs[1].set_ylabel('$C_D$')
        axs[1].set_title('Cubic Interpolation of $C_D$ vs y')
        axs[1].legend()
        axs[1].grid(True)

        # Plot Cm
        axs[2].plot(self.ylst, self.Cmlst, 'o', label='Original $C_M$', markersize=8)
        axs[2].plot(z_points, Cm_points, '-', label='Cubic Interpolation')
        axs[2].set_xlabel('y')
        axs[2].set_ylabel('$C_M$')
        axs[2].set_title('Cubic Interpolation of $C_M$ vs y')
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.show()

class InternalForces():
    def __init__(self, density, airspeed, distributions, c_r, wingspan, c_t = None, tr = None):
        """
        
        """
        self.c_t = tr * c_r if c_t is None else c_t # tip chord [m]
        self.c_r = c_r 
        self.wingspan = wingspan
        self.q = 0.5 * density * airspeed ** 2
        
        self.g_cl = distributions[0]
        self.g_cd = distributions[1]
        self.g_cm = distributions[2]
        
        self.z_points = np.linspace(0, self.wingspan / 2)
                
    # This is the main function
    def distributions(self):
        z = sp.symbols("z")
        chord_dist_z = self.c_t + ((self.c_r - self.c_t) / (0 - self.wingspan / 2)) * (z - self.wingspan / 2)
        
        # lift, torque, and drag distributions along the span
        L_z = []
        T_z = []
        D_z = []
        for i in self.z_points:
            L_z.append(self.g_cl(i) * self.q * chord_dist_z.subs(z, i))
            T_z.append(self.g_cm(i) * self.q * chord_dist_z.subs(z, i))
            D_z.append(self.g_cd(i) * self.q * chord_dist_z.subs(z, i))

        # These are the final scipy functions
        Lift_dist = interp1d(self.z_points, L_z, kind='cubic', fill_value="extrapolate")
        Torque_dist = interp1d(self.z_points, T_z, kind='cubic', fill_value="extrapolate")
        Drag_dist = interp1d(self.z_points, D_z, kind='cubic', fill_value="extrapolate")

        return Lift_dist, Torque_dist, Drag_dist
    
    def show(self):
        # subplots
        self.lift_dist, self.torque_dist, self.drag_dist = self.distributions()
        
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Lift Dist
        axs[0].plot(self.z_points, self.lift_dist(self.z_points),
                    label="Lift Distribution",
                    color="blue", linewidth=2)
        axs[0].set_xlabel("Spanwise Location (y) [m]", fontsize=12)
        axs[0].set_ylabel("Lift per Unit Span (L) [N/m]", fontsize=12)
        axs[0].set_title("Lift Distribution", fontsize=14)
        axs[0].grid(True)
        axs[0].legend(fontsize=10)

        # Torque Dist
        axs[1].plot(self.z_points, self.torque_dist(self.z_points),
                    label="Torque Distribution (Torsion)",
                    color="red", linewidth=2)
        axs[1].set_xlabel("Spanwise Location (y) [m]", fontsize=12)
        axs[1].set_ylabel("Torque per Unit Span (T) [Nm/m]", fontsize=12)
        axs[1].set_title("Torque Distribution", fontsize=14)
        axs[1].grid(True)
        axs[1].legend(fontsize=10)

        # Drag Dist
        axs[2].plot(self.z_points, self.drag_dist(self.z_points),
                    label="Drag Distribution",
                    color="green", linewidth=2)
        axs[2].set_xlabel("Spanwise Location (y) [m]", fontsize=12)
        axs[2].set_ylabel("Drag per Unit Span (D) [N/m]", fontsize=12)
        axs[2].set_title("Drag Distribution", fontsize=14)
        axs[2].grid(True)
        axs[2].legend(fontsize=10)

        plt.tight_layout()
        plt.show()
