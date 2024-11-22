import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sympy as sp

# File paths
file_name = ["venv/XFLR5sims/MainWing_a=0.00_v=10.00ms.txt", "venv/XFLR5sims/MainWing_a=1.00_v=10.00ms.txt",
             "venv/XFLR5sims/MainWing_a=2.00_v=10.00ms.txt", "venv/XFLR5sims/MainWing_a=3.00_v=10.00ms.txt",
             "venv/XFLR5sims/MainWing_a=4.00_v=10.00ms.txt", "venv/XFLR5sims/MainWing_a=5.00_v=10.00ms.txt",
             "venv/XFLR5sims/MainWing_a=6.00_v=10.00ms.txt", "venv/XFLR5sims/MainWing_a=7.00_v=10.00ms.txt",
             "venv/XFLR5sims/MainWing_a=8.00_v=10.00ms.txt", "venv/XFLR5sims/MainWing_a=9.00_v=10.00ms.txt",
             "venv/XFLR5sims/MainWing_a=10_v=10.00ms.txt"]

# AoA

# def Curves(AoA, ):

a = int(input("Angle of Attack: "))

# Reading data from XFLR5 output
ylst = np.genfromtxt(file_name[a], skip_header=40, max_rows=19, usecols=(0,), invalid_raise=False)
Cllst = np.genfromtxt(file_name[a], skip_header=40, max_rows=19, usecols=(3,), invalid_raise=False)
Cdlst = np.genfromtxt(file_name[a], skip_header=40, max_rows=19, usecols=(5,), invalid_raise=False)
Cmlst = np.genfromtxt(file_name[a], skip_header=40, max_rows=19, usecols=(7,), invalid_raise=False)

# Interpolate data
g_cl = interp1d(ylst, Cllst, kind='cubic', fill_value="extrapolate") # Cl scipy function
g_cd = interp1d(ylst, Cdlst, kind='cubic', fill_value="extrapolate") # Cd scipy function
g_cm = interp1d(ylst, Cmlst, kind='cubic', fill_value="extrapolate") # Cm scipy function

# Generate smooth interpolation points
z_points = np.linspace(2.943 / 2, 13.45, 900)
Cl_points = g_cl(z_points)
Cd_points = g_cd(z_points)
Cm_points = g_cm(z_points)

# SubPlots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot Cl
axs[0].plot(ylst, Cllst, 'o', label='Original $C_L$', markersize=8)
axs[0].plot(z_points, Cl_points, '-', label='Cubic Interpolation')
axs[0].set_xlabel('y')
axs[0].set_ylabel('$C_L$')
axs[0].set_title('Cubic Interpolation of $C_L$ vs y')
axs[0].legend()
axs[0].grid(True)

# Plot Cd
axs[1].plot(ylst, Cdlst, 'o', label='Original $C_D$', markersize=8)
axs[1].plot(z_points, Cd_points, '-', label='Cubic Interpolation')
axs[1].set_xlabel('y')
axs[1].set_ylabel('$C_D$')
axs[1].set_title('Cubic Interpolation of $C_D$ vs y')
axs[1].legend()
axs[1].grid(True)

# Plot Cm
axs[2].plot(ylst, Cmlst, 'o', label='Original $C_M$', markersize=8)
axs[2].plot(z_points, Cm_points, '-', label='Cubic Interpolation')
axs[2].set_xlabel('y')
axs[2].set_ylabel('$C_M$')
axs[2].set_title('Cubic Interpolation of $C_M$ vs y')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()


# Lift per unit span

# print(chord_dist_z)


# This is the main function
def Distributions(c_r, c_t, wingspan, density, airspeed, Cl_distribution, Cm_distribution, Cd_distribution):
    z = sp.symbols("z")
    chord_dist_z = c_t + ((c_r - c_t) / (0 - wingspan / 2)) * (z - wingspan / 2)

    # lift, torque, and drag distributions along the span
    L_z = []
    T_z = []
    D_z = []
    q = 0.5 * density * airspeed ** 2
    for i in z_points:
        L_z.append(g_cl(i) * q * chord_dist_z.subs(z, i))
        T_z.append(g_cm(i) * q * chord_dist_z.subs(z, i))
        D_z.append(g_cd(i) * q * chord_dist_z.subs(z, i))

    # These are the final scipy functions
    Lift_dist = interp1d(z_points, L_z, kind='cubic', fill_value="extrapolate")
    Torque_dist = interp1d(z_points, T_z, kind='cubic', fill_value="extrapolate")
    Drag_dist = interp1d(z_points, D_z, kind='cubic', fill_value="extrapolate")

    return Lift_dist, Torque_dist, Drag_dist


# subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Lift Dist
axs[0].plot(z_points, Distributions(4.33, 1.334, 26.9, 1.225, 66.26, Cllst, Cmlst, Cdlst)[0](z_points),
            label="Lift Distribution",
            color="blue", linewidth=2)
axs[0].set_xlabel("Spanwise Location (y) [m]", fontsize=12)
axs[0].set_ylabel("Lift per Unit Span (L) [N/m]", fontsize=12)
axs[0].set_title("Lift Distribution", fontsize=14)
axs[0].grid(True)
axs[0].legend(fontsize=10)

# Torque Dist
axs[1].plot(z_points, Distributions(4.33, 1.334, 26.9, 1.225, 66.26, Cllst, Cmlst, Cdlst)[1](z_points),
            label="Torque Distribution (Torsion)",
            color="red", linewidth=2)
axs[1].set_xlabel("Spanwise Location (y) [m]", fontsize=12)
axs[1].set_ylabel("Torque per Unit Span (T) [Nm/m]", fontsize=12)
axs[1].set_title("Torque Distribution", fontsize=14)
axs[1].grid(True)
axs[1].legend(fontsize=10)

# Drag Dist
axs[2].plot(z_points, Distributions(4.33, 1.334, 26.9, 1.225, 66.26, Cllst, Cmlst, Cdlst)[2](z_points),
            label="Drag Distribution",
            color="green", linewidth=2)
axs[2].set_xlabel("Spanwise Location (y) [m]", fontsize=12)
axs[2].set_ylabel("Drag per Unit Span (D) [N/m]", fontsize=12)
axs[2].set_title("Drag Distribution", fontsize=14)
axs[2].grid(True)
axs[2].legend(fontsize=10)

plt.tight_layout()
plt.show()
