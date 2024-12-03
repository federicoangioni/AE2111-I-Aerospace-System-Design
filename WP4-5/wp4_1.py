import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sympy as sp
import os
from scipy.integrate import cumtrapz


# authors: Medhansh, Teodor

class Aerodynamics():
    def __init__(self, folder: str, aoa: int, wingspan: int, fus_radius):
        self.files = [os.path.join(folder, file) for file in
                      os.listdir(folder)]  # makes a list with all the files in the XFLR folder
        self.aoa = aoa + 1
        self.wingspan = wingspan - 2 * fus_radius

    def coefficients(self, return_list: bool):

        ylst = np.genfromtxt(self.files[self.aoa], skip_header=40, max_rows=19, usecols=(0,), invalid_raise=False)
        Cllst = np.genfromtxt(self.files[self.aoa], skip_header=40, max_rows=19, usecols=(3,), invalid_raise=False)
        Cdlst = np.genfromtxt(self.files[self.aoa], skip_header=40, max_rows=19, usecols=(5,), invalid_raise=False)
        Cmlst = np.genfromtxt(self.files[self.aoa], skip_header=40, max_rows=19, usecols=(7,), invalid_raise=False)

        # Interpolate data
        g_cl = interp1d(ylst, Cllst, kind='cubic', fill_value="extrapolate")  # Cl scipy function, callable
        g_cd = interp1d(ylst, Cdlst, kind='cubic', fill_value="extrapolate")  # Cd scipy function, callable
        g_cm = interp1d(ylst, Cmlst, kind='cubic', fill_value="extrapolate")  # Cm scipy functio, callable

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
    def __init__(self, load_factor, sound_speed, half_chord_sweep, fus_radius, density, airspeed, distributions, c_r, wingspan, engine_z_loc, engine_length, x_hl, x_lemac, MAC, one_engine_thrust, fan_cowl_diameter, c_t):
        """

        """
        z = sp.symbols("z")
        self.c_r = c_t + ((c_r - c_t) / (0 - wingspan / 2)) * (z - wingspan / 2).subs(z, fus_radius)
        self.c_t = c_t
        self.half_chord_sweep = half_chord_sweep
        self.wingspan = wingspan - (2*fus_radius)
        self.ogwingspan = wingspan
        self.q = 0.5 * density * airspeed ** 2
        self.engine_z = engine_z_loc - fus_radius
        self.engine_length = engine_length
        self.x_hl = x_hl
        self.x_lemac = x_lemac
        self.MAC = MAC
        self.one_engine_thrust = one_engine_thrust
        self.fan_cowl_diameter = fan_cowl_diameter
        self.load_factor = load_factor
        self.fus_radius = fus_radius
        self.sound_speed = sound_speed

        self.z_points = np.linspace(0, self.wingspan / 2, 1000)

        self.g_cl = distributions[0]
        self.g_cd = distributions[1]
        self.g_cm = distributions[2]

        self.lift_dist, self.torque_dist, self.drag_dist = self.applied_distributions()  # this is ok

    # This is the main function
    def applied_distributions(self):
        z = sp.symbols("z")
        chord_dist_z = self.c_t + ((self.c_r - self.c_t) / (0 - self.wingspan / 2)) * (z - self.wingspan / 2)

        # lift, torque, and drag distributions along the span
        L_z = []
        T_z = []
        D_z = []
        for i in self.z_points:
            L_z.append(self.g_cl(i) * self.q * chord_dist_z.subs(z, i)/ np.sqrt(1-(self.airspeed/self.sound_speed)**2))
            T_z.append(self.g_cm(i) * self.q * chord_dist_z.subs(z, i)/ np.sqrt(1-(self.airspeed/self.sound_speed)**2))
            D_z.append(self.g_cd(i) * self.q * chord_dist_z.subs(z, i)/ np.sqrt(1-(self.airspeed/self.sound_speed)**2))

        # These are the final scipy functions
        Lift_dist = interp1d(self.z_points, L_z, kind='cubic', fill_value="extrapolate")
        Torque_dist = interp1d(self.z_points, T_z, kind='cubic', fill_value="extrapolate")
        Drag_dist = interp1d(self.z_points, D_z, kind='cubic', fill_value="extrapolate")

        return Lift_dist, Torque_dist, Drag_dist

    def internal_force_diagrams(self, engine_mass, wing_box_length, fuel_tank_length, fuel_density, return_list = False):
        z = sp.symbols("z")
        
        # Engine weight
        eng_weight = - engine_mass * 9.81 * self.load_factor

        # Wing weight distribution
        # I assumed that the net wing weight will act at 25% of the half wingspan
        Wing_weight_distribution = -(-(141.1 * self.load_factor * z) + (1897.8 * self.load_factor))
        Total_Wing_weight_force = sp.integrate(Wing_weight_distribution, (z, self.fus_radius, self.ogwingspan/2))
        Total_Wing_weight_force_z_loc = sp.integrate(z * Wing_weight_distribution, (z, self.fus_radius, self.ogwingspan/2)) / sp.integrate(
            Wing_weight_distribution, (z, self.fus_radius, self.ogwingspan/2))
        Wing_struc_shear = sp.integrate(Wing_weight_distribution, (z, self.fus_radius, z))

        # Fuel weight distribution -> NEEDS TO BE ADJUSTED
        # I'm assuming that the fuel is distributed throughout the entire wing (it is
        # significantly higher than the required fuel as well)
        A_root = (self.c_r * wing_box_length) * (self.c_r * 0.14)  # Im using wingbox areas at the root and tip chord
        A_tip = (self.c_t * wing_box_length) * (self.c_t * 0.14)
        Fuel_weight_distribution = -(A_root - ((A_root - A_tip) / fuel_tank_length) * z) * fuel_density * self.load_factor
        Total_Fuel_force = (sp.integrate(Fuel_weight_distribution, (z, 0, fuel_tank_length)))
        Total_Fuel_force_z_loc = sp.integrate(z * Fuel_weight_distribution, (z, 0, fuel_tank_length)) / sp.integrate(
            Fuel_weight_distribution, (z, 0, fuel_tank_length))
        Fuel_shear = sp.integrate(Fuel_weight_distribution, (z, 0, z))

        # Aerodynamic load
        coefficients = np.polyfit(self.z_points, self.lift_dist(self.z_points),
                                  deg=16)  # Co-efficients for polynomial approximation of lift
        Lift_load_distribution = sum(c * z ** i for i, c in enumerate(reversed(coefficients)))
        Total_lift_force = sp.integrate(Lift_load_distribution, (z, 0, self.wingspan / 2))
        Total_lift_force_z_loc = sp.integrate(z * Lift_load_distribution, (z, 0, self.wingspan / 2)) / sp.integrate(
            Lift_load_distribution, (z, 0, self.wingspan / 2))
        Lift_shear = sp.integrate(Lift_load_distribution, (z, 0, z))

        # Torque distribution
        lift_torque_coefficients = np.polyfit(self.z_points, self.torque_dist(self.z_points),
                                              deg=16)  # Co-efficients for polynomial approximation of torque
        lift_torque_load_distribution = sum(c * z ** i for i, c in enumerate(reversed(lift_torque_coefficients)))
        total_lift_torque = sp.integrate(lift_torque_load_distribution, (z, 0, self.wingspan / 2))
        lift_torque = sp.integrate(lift_torque_load_distribution, (z, 0, z))
        l1 = self.engine_length + self.x_hl
        l2 = self.x_lemac + 0.5 * self.MAC
        x_eng = l2 - l1
        eng_torque = eng_weight * x_eng + self.one_engine_thrust * np.cos(self.half_chord_sweep * np.pi/180) * (self.fan_cowl_diameter / 2)

        # Reactions
        reaction_force = -(Total_Fuel_force + Total_lift_force + Total_Wing_weight_force + eng_weight)
        reaction_moment = (
                (self.one_engine_thrust * np.sin(self.half_chord_sweep * np.pi/180) * self.fan_cowl_diameter/2) + Total_Fuel_force * Total_Fuel_force_z_loc + Total_lift_force * Total_lift_force_z_loc + Total_Wing_weight_force * Total_Wing_weight_force_z_loc + eng_weight * self.engine_z)
        reaction_torque = -(total_lift_torque + eng_torque)
        reaction_axial_force = - self.one_engine_thrust * np.sin(self.half_chord_sweep * np.pi/180)


        shear_distribution = []

        for i in self.z_points:
            if i < self.engine_z:
                shear_value = (reaction_force + Lift_shear + Wing_struc_shear + Fuel_shear).subs(z, i).evalf()
            else:
                shear_value = (reaction_force + Lift_shear + Wing_struc_shear + Fuel_shear + eng_weight).subs(z,
                                                                                                              i).evalf()

            shear_distribution.append(shear_value)

        shear_distribution = np.array(shear_distribution, dtype=float)


        moment_distribution = cumtrapz(shear_distribution, self.z_points, initial=0)
        moment_distribution = np.array(moment_distribution, dtype=float)
        moment_distribution = moment_distribution + reaction_moment


        torque_distribution = []
        axial_distribution = []

        for i in self.z_points:
            if i < self.engine_z:
                torque_value = (reaction_torque + lift_torque).subs(z, i).evalf()
                axial_value = reaction_axial_force
            else:
                torque_value = (reaction_torque + lift_torque + eng_torque).subs(z, i).evalf()
                axial_value = 0

            torque_distribution.append(torque_value)
            axial_distribution.append((axial_value))


        shear_list = list(shear_distribution)
        moment_list1 = list(moment_distribution)
        torque_list = list(torque_distribution)
        axial_list = list(axial_distribution)

        moment_list = []
        idx = 0

        for i in self.z_points:
            if i < self.engine_z:
                moment_list.append(moment_list1[idx])
            else:
                moment_list.append(moment_list1[idx] - self.one_engine_thrust * np.sin(self.half_chord_sweep * np.pi/180) * self.fan_cowl_diameter/2)
            idx += 1

        g_shear = interp1d(self.z_points, shear_list, kind='cubic', fill_value="extrapolate")
        g_moment = interp1d(self.z_points, moment_list, kind='cubic', fill_value="extrapolate")
        g_torque = interp1d(self.z_points, torque_list, kind='cubic', fill_value="extrapolate")
        g_axial = interp1d(self.z_points, axial_list, kind='cubic', fill_value="extrapolate")
        # shear list is okay
        if return_list:
            return shear_list, moment_list, torque_list, axial_list
        else:
            return g_shear, g_moment, g_torque, g_axial

    def show(self, engine_mass, wing_box_length, fuel_tank_length, fuel_density):
        # subplots
        shear, moment, torque, axial_force = self.internal_force_diagrams(engine_mass=engine_mass, wing_box_length=wing_box_length,
                                                               fuel_tank_length=fuel_tank_length,
                                                               fuel_density=fuel_density, return_list = True)
        shear = [0] + shear
        moment = [0] + moment
        torque = [0] + torque
        axial_force = [0] + axial_force

        self.z_points = [0] + list(self.z_points)

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

        plt.show()
        plt.close()
        
        fig, axs = plt.subplots(1, 4, figsize=(18, 6))
        # Plot Load Distribution
    
        axs[0].plot(self.z_points, shear, label='Shear Force Distribution', color='blue', linewidth=2)
        axs[0].axhline(0, color="black", linewidth=0.8)
        axs[0].set_xlabel("Span-wise Location (z) [m]", fontsize=12)
        axs[0].set_ylabel("Shear Force [N]", fontsize=12)
        axs[0].set_title("Shear Force Distribution Along the Wing Span", fontsize=14)
        axs[0].grid(True)
        axs[0].legend(fontsize=10)
        
        
        # Plot moment Distribution
        
        axs[1].plot(self.z_points, moment, label='Bending Moment Distribution', color='blue', linewidth=2)
        axs[1].axhline(0, color="black", linewidth=0.8)
        axs[1].set_xlabel("Span-wise Location (z) [m]", fontsize=12)
        axs[1].set_ylabel("Bending Moment [kNm]", fontsize=12)
        axs[1].set_title("Moment Distribution Along the Wing Span", fontsize=14)
        axs[1].grid(True)
        axs[1].legend(fontsize=10)
        

        # Plot torque Distribution
        
        axs[2].plot(self.z_points, torque, label='Torque Distribution', color='blue', linewidth=2)
        axs[2].axhline(0, color="black", linewidth=0.8)
        axs[2].set_xlabel("Span-wise Location (z) [m]", fontsize=12)
        axs[2].set_ylabel("Torque [Nm]", fontsize=12)
        axs[2].set_title("Torque Distribution Along the Wing Span", fontsize=14)
        axs[2].grid(True)
        axs[2].legend(fontsize=10)
        

        # Plot axial force Distribution
        
        axs[3].plot(self.z_points, axial_force, label='Axial Force Distribution', color='blue', linewidth=2)
        axs[3].axhline(0, color="black", linewidth=0.8)
        axs[3].set_xlabel("Span-wise Location (z) [m]", fontsize=12)
        axs[3].set_ylabel("Axial Force [N]", fontsize=12)
        axs[3].set_title("Axial Force Distribution Along the Wing Span", fontsize=14)
        axs[3].grid(True)
        axs[3].legend(fontsize=10)
        plt.tight_layout()
        plt.show()

# change
# aerodynamics = Aerodynamics(folder="C:\\Users\medha\OneDrive\Desktop\TU Delft 1st Year\pythonProject1\\venv\XFLROpPoints2", aoa=5, wingspan=26.9, fus_radius=1.47)
# distributions = aerodynamics.coefficients(return_list=False)

# internal_forces = InternalForces(
#     load_factor= 1,
#     sound_speed=343,
#     half_chord_sweep=22.4645,
#     fus_radius=1.47,
#     density=1.225,            # Air density in kg/m^3
#     airspeed=66.26,               # Airspeed in m/s
#     distributions=distributions,  # From the Aerodynamics class
#     c_r=4.33,                 # Root chord in meters
#     c_t=1.33,
#     wingspan= 26.9,            # Wingspan in meters after shaving off fuselage width
#     engine_z_loc=4.35,        # Engine location along the span (z) in meters
#     engine_length=3.0,        # Length of the engine in meters
#     x_hl=13,                  # Horizontal distance to the engine
#     x_lemac=16.37,            # Horizontal distance to the leading edge of MAC
#     MAC=3.05,                 # Mean Aerodynamic Chord in meters
#     one_engine_thrust=78466,  # Thrust per engine in Newtons
#     fan_cowl_diameter=1.448,  # Diameter of the fan cowl in meters        
# )

# internal_forces.show(
#     engine_mass=2306,         # Mass of the engine in kilograms
#     wing_box_length=0.55,     # Length of the wingbox relative to chord
#     fuel_tank_length=11.98,   # Length of the fuel tank in meters
#     fuel_density=800        # Density of the fuel in kg/m^3
# )

# def critical_case_analysis(aoa, load_factor, airspeed, density, one_engine_thrust):
#     aerodynamics = Aerodynamics(
#         folder="XFLRdata\XFLR5sims", aoa=aoa,
#         wingspan=26.9, fus_radius=1.47)
#     distributions = aerodynamics.coefficients(return_list=False)

#     internal_forces = InternalForces(
#         load_factor=load_factor,
#         sound_speed=343,
#         half_chord_sweep=22.4645,
#         fus_radius=1.47,
#         density=density,  # Air density in kg/m^3
#         airspeed=airspeed,  # Airspeed in m/s
#         distributions=distributions,  # From the Aerodynamics class
#         c_r=4.33,  # Root chord in meters
#         c_t=1.33,
#         wingspan=26.9,  # Wingspan in meters after shaving off fuselage width
#         engine_z_loc=4.35,  # Engine location along the span (z) in meters
#         engine_length=3.0,  # Length of the engine in meters
#         x_hl=13,  # Horizontal distance to the engine
#         x_lemac=16.37,  # Horizontal distance to the leading edge of MAC
#         MAC=3.05,  # Mean Aerodynamic Chord in meters
#         one_engine_thrust=one_engine_thrust,  # Thrust per engine in Newtons
#         fan_cowl_diameter=1.448,  # Diameter of the fan cowl in meters
#     )

#     internal_forces.show(
#         engine_mass=2306,  # Mass of the engine in kilograms
#         wing_box_length=0.55,  # Length of the wingbox relative to chord
#         fuel_tank_length=11.98,  # Length of the fuel tank in meters
#         fuel_density=800  # Density of the fuel in kg/m^3
#     )
    
# # critical_case_analysis(aoa= 10, load_factor= 2.5, airspeed= 327.5, density= 1.225, one_engine_thrust= 78466)
