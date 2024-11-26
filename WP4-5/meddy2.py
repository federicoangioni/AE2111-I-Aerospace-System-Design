import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import sympy as sp
from scipy import integrate

# File paths
file_name = ["XFLRdata/XFLR5sims/MainWing_a=0.00_v=10.00ms.txt", "XFLRdata/XFLR5sims/MainWing_a=1.00_v=10.00ms.txt",
             "XFLRdata/XFLR5sims/MainWing_a=2.00_v=10.00ms.txt", "XFLRdata/XFLR5sims/MainWing_a=3.00_v=10.00ms.txt",
             "XFLRdata/XFLR5sims/MainWing_a=4.00_v=10.00ms.txt", "XFLRdata/XFLR5sims/MainWing_a=5.00_v=10.00ms.txt",
             "XFLRdata/XFLR5sims/MainWing_a=6.00_v=10.00ms.txt", "XFLRdata/XFLR5sims/MainWing_a=7.00_v=10.00ms.txt",
             "XFLRdata/XFLR5sims/MainWing_a=8.00_v=10.00ms.txt", "XFLRdata/XFLR5sims/MainWing_a=9.00_v=10.00ms.txt",
             "XFLRdata/XFLR5sims/MainWing_a=10_v=10.00ms.txt"]

a = int(input("Angle of Attack: "))

# Reading data from XFLR5 output
ylst = np.genfromtxt(file_name[a], skip_header=40, max_rows=19, usecols=(0,), invalid_raise=False)
Cllst = np.genfromtxt(file_name[a], skip_header=40, max_rows=19, usecols=(3,), invalid_raise=False)
Cdlst = np.genfromtxt(file_name[a], skip_header=40, max_rows=19, usecols=(5,), invalid_raise=False)
Cmlst = np.genfromtxt(file_name[a], skip_header=40, max_rows=19, usecols=(7,), invalid_raise=False)

# Interpolate data
g_cl = interp1d(ylst, Cllst, kind='cubic', fill_value="extrapolate")  # Cl scipy function
g_cd = interp1d(ylst, Cdlst, kind='cubic', fill_value="extrapolate")  # Cd scipy function
g_cm = interp1d(ylst, Cmlst, kind='cubic', fill_value="extrapolate")  # Cm scipy function

# Generate smooth interpolation points
z_points = np.linspace(0, 13.45, 900)
Cl_points = g_cl(z_points)
Cd_points = g_cd(z_points)
Cm_points = g_cm(z_points)


# # SubPlots
# fig, axs = plt.subplots(1, 3, figsize=(18, 6))
#
# # Plot Cl
# axs[0].plot(ylst, Cllst, 'o', label='Original $C_L$', markersize=8)
# axs[0].plot(z_points, Cl_points, '-', label='Cubic Interpolation')
# axs[0].set_xlabel('y')
# axs[0].set_ylabel('$C_L$')
# axs[0].set_title('Cubic Interpolation of $C_L$ vs y')
# axs[0].legend()
# axs[0].grid(True)
#
# # Plot Cd
# axs[1].plot(ylst, Cdlst, 'o', label='Original $C_D$', markersize=8)
# axs[1].plot(z_points, Cd_points, '-', label='Cubic Interpolation')
# axs[1].set_xlabel('y')
# axs[1].set_ylabel('$C_D$')
# axs[1].set_title('Cubic Interpolation of $C_D$ vs y')
# axs[1].legend()
# axs[1].grid(True)
#
# # Plot Cm
# axs[2].plot(ylst, Cmlst, 'o', label='Original $C_M$', markersize=8)
# axs[2].plot(z_points, Cm_points, '-', label='Cubic Interpolation')
# axs[2].set_xlabel('y')
# axs[2].set_ylabel('$C_M$')
# axs[2].set_title('Cubic Interpolation of $C_M$ vs y')
# axs[2].legend()
# axs[2].grid(True)
#
# plt.tight_layout()
# plt.show()
#

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

    return Lift_dist, Torque_dist, Drag_dist, L_z, T_z, D_z


# print(Distributions(4.33, 1.334, 26.9, 1.225, 66.26, Cllst, Cmlst, Cdlst)[0](2))


# # subplots
# fig, axs = plt.subplots(1, 3, figsize=(18, 6))
#
# # Lift Dist
# axs[0].plot(z_points, Distributions(4.33, 1.334, 26.9, 1.225, 66.26, Cllst, Cmlst, Cdlst)[0](z_points),
#             label="Lift Distribution",
#             color="blue", linewidth=2)
# axs[0].set_xlabel("Span-wise Location (y) [m]", fontsize=12)
# axs[0].set_ylabel("Lift per Unit Span (L) [N/m]", fontsize=12)
# axs[0].set_title("Lift Distribution", fontsize=14)
# axs[0].grid(True)
# axs[0].legend(fontsize=10)
#
# # Torque Dist
# axs[1].plot(z_points, Distributions(4.33, 1.334, 26.9, 1.225, 66.26, Cllst, Cmlst, Cdlst)[1](z_points),
#             label="Torque Distribution",
#             color="red", linewidth=2)
# axs[1].set_xlabel("Span-wise Location (y) [m]", fontsize=12)
# axs[1].set_ylabel("Torque per Unit Span (T) [Nm/m]", fontsize=12)
# axs[1].set_title("Torque Distribution", fontsize=14)
# axs[1].grid(True)
# axs[1].legend(fontsize=10)
#
# # Drag Dist
# axs[2].plot(z_points, Distributions(4.33, 1.334, 26.9, 1.225, 66.26, Cllst, Cmlst, Cdlst)[2](z_points),
#             label="Drag Distribution",
#             color="green", linewidth=2)
# axs[2].set_xlabel("Span-wise Location (y) [m]", fontsize=12)
# axs[2].set_ylabel("Drag per Unit Span (D) [N/m]", fontsize=12)
# axs[2].set_title("Drag Distribution", fontsize=14)
# axs[2].grid(True)
# axs[2].legend(fontsize=10)
#
# plt.tight_layout()
# plt.show()


# Weight distribution function

# Shear Force Calculation
def Force_diagrams(wing_span, engine_mass, engine_z_loc, c_r, c_t, wing_box_minimum_width_to_chord_ratio, fuel_tank_length, fuel_density, L_z):
    z = sp.symbols("z")
    span_sections = np.linspace(0, wing_span / 2, 900)

    # Engine weight
    eng_weight = -engine_mass * 9.81

    # Wing weight distribution
    # I assumed that the net wing weight will act at 25% of the half wingspan
    Wing_weight_distribution = -(-211.69 * z + 2372.62)
    Total_Wing_weight_force = sp.integrate(Wing_weight_distribution, (z, 0, wing_span / 2))
    Total_Wing_weight_force_z_loc = 0.25 * wing_span / 2
    Wing_struc_shear = sp.integrate(Wing_weight_distribution, (z, 0, z))

    # Fuel weight distribution -> NEEDS TO BE ADJUSTED
    # I'm assuming that the fuel is distributed throughout the entire wing (it is
    # significantly higher than the required fuel as well)
    A_root = (c_r * wing_box_minimum_width_to_chord_ratio) * (c_r * 0.14) # Im using wingbox areas at the root and tip chord
    A_tip = (c_t * wing_box_minimum_width_to_chord_ratio) * (c_t * 0.14)
    Fuel_weight_distribution = -(A_root - ((A_root - A_tip) / fuel_tank_length) * z) * fuel_density
    Total_Fuel_force = (sp.integrate(Fuel_weight_distribution, (z, 0, fuel_tank_length)))
    Total_Fuel_force_z_loc = sp.integrate(z * Fuel_weight_distribution, (z, 0, fuel_tank_length)) / sp.integrate(Fuel_weight_distribution, (z, 0, fuel_tank_length))
    Fuel_shear = sp.integrate(Fuel_weight_distribution, (z, 0, z))

    # Aerodynamic load
    coefficients = np.polyfit(z_points, L_z, deg=16)  # Co-efficients for polynomial approximation of lift
    Lift_load_distribution = sum(c * z ** i for i, c in enumerate(reversed(coefficients)))
    Total_lift_force = sp.integrate(Lift_load_distribution, (z, 0, wing_span / 2))
    Total_lift_force_z_loc = sp.integrate(z * Lift_load_distribution, (z, 0, wing_span / 2)) / sp.integrate(Lift_load_distribution, (z, 0, wing_span / 2))
    Lift_shear = sp.integrate(Lift_load_distribution, (z, 0, z))

    # Reactions
    reaction_force = -(Total_Fuel_force + Total_lift_force + Total_Wing_weight_force + eng_weight)
    reaction_moment = (Total_Fuel_force * Total_Fuel_force_z_loc + Total_lift_force * Total_lift_force_z_loc + Total_Wing_weight_force * Total_Wing_weight_force_z_loc + eng_weight * engine_z_loc)

    shear_distribution = []

    for i in span_sections:
        if i < engine_z_loc:
            shear_value = (reaction_force + Lift_shear + Wing_struc_shear + Fuel_shear).subs(z, i).evalf()
        else:
            shear_value = (reaction_force + Lift_shear + Wing_struc_shear + Fuel_shear + eng_weight).subs(z, i).evalf()

        shear_distribution.append(shear_value)

    shear_distribution = np.array(shear_distribution, dtype=float)

    moment_distribution = cumtrapz(shear_distribution, span_sections, initial=0)
    moment_distribution = np.array(moment_distribution, dtype=float)
    moment_distribution = moment_distribution + reaction_moment

    return [0] + list(span_sections), [0] + list(shear_distribution), [0] + list(moment_distribution), engine_z_loc


# Calculate shear
shear = Force_diagrams(26.9, 2306, 4.351, 4.33, 1.33, 0.6, 13.45, 800, Distributions(4.33, 1.334, 26.9, 1.225, 66.26, Cllst, Cmlst, Cdlst)[0](z_points))

# Plot Load Distribution
plt.figure(figsize=(12, 6))
plt.plot(shear[0], shear[1], label='Shear Force Distribution', color='blue', linewidth=2)
plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Span-wise Location (z) [m]", fontsize=12)
plt.ylabel("Shear Force [N]", fontsize=12)
plt.title("Shear Force Distribution Along the Wing Span", fontsize=14)
plt.grid(True)
plt.legend(fontsize=10)
plt.show()

# Plot moment Distribution
plt.figure(figsize=(12, 6))
plt.plot(shear[0], shear[2], label='Bending Moment Distribution', color='blue', linewidth=2)
plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Span-wise Location (z) [m]", fontsize=12)
plt.ylabel("Bending Moment [kNm]", fontsize=12)
plt.title("Moment Distribution Along the Wing Span", fontsize=14)
plt.grid(True)
plt.legend(fontsize=10)
plt.show()
