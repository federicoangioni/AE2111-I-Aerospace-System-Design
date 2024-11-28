from matplotlib import pyplot as plt
import numpy as np
import math
from ISA import AtmosphericConditions as ISA
import variables as var
import os


def Prandtl_Glauert_correction(Mach, CL):
    return CL / math.sqrt(1 - Mach ** 2)     


# Speeds
def speed_from_lift(Lift, rho, S, Cl):
    return math.sqrt((2 * Lift) / (rho * S * Cl))

def dive_speed_formula(speed_cruise, factor = 1.25):
    return speed_cruise * factor

def design_flap_speed(MLW, MTOW, density, wing_area, CL_max_flapped, CL_max_clean):
    opt1 = 1.6 * speed_from_lift(MTOW, density, wing_area, CL_max_flapped) 
    opt2 = 1.8 * speed_from_lift(MLW, density, wing_area, CL_max_clean)
    opt3 = 1.8 * speed_from_lift(MLW, density, wing_area, CL_max_flapped)
    return max(opt1, opt2, opt3)

def n_stall_speed(speed, stall_speed):
    return (speed / stall_speed) ** 2

def n_linear_lower_part(speed, cruise_speed, dive_speed):
    return (1 / (dive_speed - cruise_speed)) * (speed - dive_speed)


# Points
def V_n_line_upper_clean(speeds, n_max, speed_stall_clean, speed_cruise):
    dive_speed = dive_speed_formula(speed_cruise)
    list = [
        n_stall_speed(speed, speed_stall_clean) if n_stall_speed(speed, speed_stall_clean) <= n_max 
        else n_max if speed <= dive_speed 
        else 0 
        for speed in speeds
    ]
    return np.array(list)

def V_n_line_lower_clean(speeds, n_min, speed_stall_clean, speed_cruise):
    dive_speed = dive_speed_formula(speed_cruise)
    list = [
        -n_stall_speed(speed, speed_stall_clean) if -n_stall_speed(speed, speed_stall_clean) >= n_min 
        else n_min if speed <= speed_cruise 
        else n_linear_lower_part(speed, speed_cruise, dive_speed) 
        for speed in speeds
    ]
    return np.array(list)

def V_n_line_upper_flaps(speeds, n_max_flaps, speed_stall_flaps, speed_design_flap):
    list = [
        n_stall_speed(speed, speed_stall_flaps) if n_stall_speed(speed, speed_stall_flaps) <= n_max_flaps 
        else n_max_flaps  
        for speed in speeds 
        if speed <= speed_design_flap
    ]
    return np.array(list)


    
class VelocityLoadFactorDiagram():
    def __init__(self, 
            weight_kg: float, 
            MLW_kg: float,
            MTOW_kg: float,
            altitude: float, 
            CL_max_clean: float, 
            CL_max_flapped: float, 
            wing_area = 71.57, 
            cruise_mach = 0.77,
            LE_sweep = 27.7
            ):
        
        self.weight_kg = weight_kg
        self.weight = weight_kg * var.g
        self.MLW_kg = MLW_kg
        self.MLW = MLW_kg * var.g
        self.MTOW_kg = MTOW_kg
        self.MTOW = MTOW_kg * var.g
        self.CL_max_clean = CL_max_clean
        self.CL_max_flapped = CL_max_flapped
        self.n_min = -1
        self.n_max_flapped = 2
        self.wing_area = wing_area
        self.LE_sweep = LE_sweep
        self.altitude = altitude
        Atm = ISA(altitude)
        self.density = Atm.get_density()
        self.temperature = Atm.get_temperature()
        self.speed_of_sound = Atm.get_speed_of_sound()
        self.cruise_mach = cruise_mach
        self.V_cr = cruise_mach * self.speed_of_sound
        self.generate_points()

    def generate_points(self, number_of_points = 10000):
        self.max_n = max_n = self.n_max_formula()
        weight = self.weight
        density = self.density
        wing_area = self.wing_area
        V_cr = self.V_cr
        n_min = self.n_min
        n_max_flapped = self.n_max_flapped

        CL_max_clean = Prandtl_Glauert_correction(self.cruise_mach * math.cos(math.radians(self.LE_sweep)), self.CL_max_clean)
        CL_max_flapped = Prandtl_Glauert_correction(self.cruise_mach * math.cos(math.radians(self.LE_sweep)), self.CL_max_flapped)

        speed_stall_clean = speed_from_lift(weight, density, wing_area, CL_max_clean)
        speed_stall_flaps = speed_from_lift(weight, density, wing_area, CL_max_flapped)
        speed_design_flap = design_flap_speed(self.MLW, self.MTOW, density, wing_area, CL_max_flapped, CL_max_clean)

        self.speed_stall_clean = speed_stall_clean
        self.speed_stall_flaps = speed_stall_flaps
        self.speed_design_flap = speed_design_flap
        
        speeds = np.linspace(0, dive_speed_formula(V_cr), number_of_points)
        n_upper_clean = V_n_line_upper_clean(speeds, max_n, speed_stall_clean, V_cr)
        n_lower_clean = V_n_line_lower_clean(speeds, n_min, speed_stall_clean, V_cr)
        n_upper_flaps = V_n_line_upper_flaps(speeds, n_max_flapped, speed_stall_flaps, speed_design_flap)

        speeds = np.append(speeds, dive_speed_formula(V_cr))
        n_upper_clean = np.append(n_upper_clean, 0)
        n_lower_clean = np.append(n_lower_clean, 0)

        self.speeds = speeds
        self.n_upper_clean = n_upper_clean
        self.n_lower_clean = n_lower_clean
        self.n_upper_flaps = n_upper_flaps
    
    def show(self):
        speeds = self.speeds
        n_upper_clean = self.n_upper_clean
        n_lower_clean = self.n_lower_clean
        n_upper_flaps = self.n_upper_flaps

        plt.plot(speeds, n_lower_clean)
        plt.plot(speeds, n_upper_clean)
        plt.plot(speeds[:len(n_upper_flaps)], n_upper_flaps)
        plt.grid()
        plt.xlabel("Equivalent air speed [m/s]")
        plt.ylabel("Load factor")
        plt.title(f"V-n diagram altitude: {self.altitude}m, weight: {self.weight_kg}kg")
        plt.show()
    
    def save(self, path: str):
        speeds = self.speeds
        n_upper_clean = self.n_upper_clean
        n_lower_clean = self.n_lower_clean
        n_upper_flaps = self.n_upper_flaps

        plt.plot(speeds, n_lower_clean)
        plt.plot(speeds, n_upper_clean)
        plt.plot(speeds[:len(n_upper_flaps)], n_upper_flaps)
        plt.grid()
        plt.xlabel("Equivalent air speed [m/s]")
        plt.ylabel("Load factor")
        plt.title(f"V-n diagram altitude: {self.altitude}m, weight: {self.weight_kg}kg")
        plt.savefig(path, dpi=300)

    def n_max_formula(self):
        weight_kg = self.MTOW_kg
        kg_to_lbs = 2.20462
        n_max_formula = 2.1 + (2400 / ((kg_to_lbs * weight_kg) + 1000))
        return min(2.5,max(n_max_formula, 3.8))
    
    def get_speeds(self):
        return self.speeds
    
    def get_n_upper_clean(self):
        return self.n_upper_clean
    
    def get_n_lower_clean(self):
        return self.n_lower_clean
    
    def get_n_upper_flaps(self):
        return self.n_upper_flaps
    
    def get_weight(self):
        return self.weight
    
    def get_stall_speeds(self):
        return self.speed_stall_clean, self.speed_stall_flaps
    
    def get_cruise_speed(self):
        return self.V_cr
    
    def get_dive_speed(self):
        return dive_speed_formula(self.V_cr)
    
    def get_max_n_flaps(self):
        return self.n_max_flapped
    
    def get_max_n_clean(self):
        return self.max_n
    
    def get_min_n(self):
        return self.n_min
    
    def get_altitude(self):
        return self.altitude
    
    def get_speed_design_flap(self):
        return self.speed_design_flap
    
    def get_speed_manouvering(self):
        return self.speed_stall_clean * math.sqrt(self.max_n)

class LoadCases():
    def __init__(self, Vn: VelocityLoadFactorDiagram):
        self.Vn = Vn
        self.speeds = Vn.get_speeds()
        self.n_upper_clean = Vn.get_n_upper_clean()
        self.n_lower_clean = Vn.get_n_lower_clean()
        self.n_upper_flaps = Vn.get_n_upper_flaps()

        self.max_n_flaps = Vn.get_max_n_flaps()
        self.max_n_clean = Vn.get_max_n_clean()
        self.min_n = Vn.get_min_n()

        self.speed_stall_clean, self.speed_stall_flap = Vn.get_stall_speeds()
        self.speed_cruise = Vn.get_cruise_speed()
        self.speed_dive = Vn.get_dive_speed()
        self.speed_design_flap = Vn.get_speed_design_flap()
        self.speed_manouvering = Vn.get_speed_manouvering()

        self.flaps_infliction_point = self.speed_stall_flap * math.sqrt(self.max_n_flaps)
        self.clean_upper_infliction_point = self.speed_stall_clean * math.sqrt(self.max_n_clean)
        self.clean_lower_infliction_point = self.speed_stall_clean * math.sqrt(abs(self.min_n))
        
        
        self.critical_load_cases = np.array([
            self.case_given_speed_given_n(self.speed_stall_clean, flaps="TO", description="Stall Speed clean at n=1", label="VS1N1"), # stall speed n = 1
            self.case_given_speed_given_n(self.clean_upper_infliction_point, description="Manouvering speed at n=1", label="VAN1"), # manouvering speed n = 1
            self.case_given_speed_given_n(self.speed_dive, description="Dive speed at n=1", label="VDN1"), # dive speed n = 1
            self.case_last_of_line(self.n_upper_clean, description="Dive speed", label="VD"), # clean upper dive speed
            self.case_last_of_line(self.n_upper_flaps, flaps="TO", description="Flap design speed", label="VF"), # flap design speed (n=2?)
            self.case_given_speed(self.flaps_infliction_point, self.n_upper_flaps, flaps="TO", description="Flap inflection point", label="IPF"), # flap inflection point (n=2)
            self.case_given_speed(self.clean_upper_infliction_point, self.n_upper_clean, description="Manouvering speed", label="VA"), # clean upper inflection point(manouvering speed)
            self.case_given_speed(self.speed_cruise, self.n_lower_clean, description="Cruise Speed negative load", label="VCNeg", left_value=1), # Clean lower cruise speed
            self.case_given_speed(self.clean_lower_infliction_point, self.n_lower_clean, description="Inflection point negative load", label="IPNeg") # Clean lower inflection point (n=-1)
        ])

    def case_given_speed_given_n(self, speed: float, n=1., flaps = "Clean", description = "", label = ""):
        speed = self.speeds[np.where(self.speeds >= speed)[0][0]]
        load_factor = n
        weight = self.Vn.get_weight()
        return {"load_factor": load_factor, "speed": speed, "weight": weight, "flaps": flaps, "altitude": self.Vn.get_altitude(), "description": description}
    
    def case_given_speed(self, speed: float, line: np.ndarray, flaps = "Clean", description = "", label = "", left_value = 0):
        speed = self.speeds[np.where(self.speeds >= speed - left_value)[0][0]]
        load_factor = line[np.where(self.speeds >= speed - left_value)[0][0]]
        weight = self.Vn.get_weight()
        return {"load_factor": load_factor, "speed": speed, "weight": weight, "flaps": flaps, "altitude": self.Vn.get_altitude(), "description": description}
    
    def case_last_of_line(self, line: np.ndarray, flaps = "Clean", description = "", label = ""):
        load_factor = line[-2]
        speed = self.speeds[len(line) - 2]
        weight = self.Vn.get_weight()
        return {"load_factor": load_factor, "speed": speed, "weight": weight, "flaps": flaps, "altitude": self.Vn.get_altitude(), "description": description}
    
    def show(self, save = False, show=True):
        speeds = self.speeds
        n_upper_clean = self.n_upper_clean
        n_lower_clean = self.n_lower_clean
        n_upper_flaps = self.n_upper_flaps

        plt.figure(figsize=(8,5))
        plt.plot(speeds, n_lower_clean)
        plt.plot(speeds, n_upper_clean)
        plt.plot(speeds[:len(n_upper_flaps)], n_upper_flaps)
        
        plt.axhline(1, color="grey", ls="--")
        plt.axhline(0, color="black", linewidth=1.5)
        
        for case in self.critical_load_cases:
            plt.scatter(case["speed"], case["load_factor"], color="black", zorder=5)
        
        plt.plot([self.speed_stall_flap,self.speed_stall_flap],[0,1], ls=":", color="grey")
        plt.plot([self.speed_stall_clean,self.speed_stall_clean],[0,1], ls=":", color="grey")
        plt.plot([self.speed_cruise,self.speed_cruise],[self.min_n,self.max_n_clean], ls=":", color="grey")
        plt.plot([self.speed_design_flap,self.speed_design_flap],[0,self.max_n_flaps], ls=":", color="green")
        plt.plot([self.speed_manouvering,self.speed_manouvering],[0,self.max_n_clean], ls=":", color="grey")

        under = -0.15
        over = 0.1
        side = 10
        size = 8

        plt.text(self.speed_stall_flap, under, "VS0", color="black", ha="center", fontsize=size)
        plt.text(self.speed_stall_clean, 2*under, "VS1", color="black", ha="center", fontsize=size)
        plt.text(self.speed_design_flap, under, "VF", color="black", ha="center", fontsize=size)
        plt.text(self.speed_manouvering, 2*under, "VA", color="black", ha="center", fontsize=size)
        plt.text(self.speed_cruise - side, under, "VC", color="black", ha="center", fontsize=size)
        plt.text(self.speed_dive - side, over, "VD", color="black", ha="center", fontsize=size)

        plt.xlabel("Equivalent air speed [m/s]")
        plt.ylabel("Load factor")
        plt.title(f"V-n diagram altitude: {round(self.Vn.get_altitude(),2)}m, weight: {round(self.Vn.get_weight(),2)}kg")
        if save:
            os.makedirs("VNDiagram", exist_ok=True)
            plt.savefig(f"VNDiagram/W{round(self.Vn.get_weight())}_A{round(self.Vn.get_altitude())}.png", dpi=300)
        if show:
            plt.show()


    def get_load_cases(self):
        return self.critical_load_cases
    

if __name__ == "__main__":
    print("running wp4-3.py")
    MTOW_kg = 19593  #kg
    weights_kg = [19593, 19593+6355, 35688] # OEW | OEW + MPW | OEW + MPW + Fuel (AKA MTOW)
    altitudes_m = [0, 35000 * 0.3048]
    critical_cases = np.empty(0)
    CL_max_clean = 1.41
    CL_max_flapped = 2.55

    for altitude in altitudes_m:
        for weight in weights_kg:
            VND = VelocityLoadFactorDiagram(weight, 0.886*weights_kg[2], weights_kg[2], altitude, CL_max_clean, CL_max_flapped)

            LC = LoadCases(VND)
            LC.show(show=False, save=True)
            critical_cases = np.append(critical_cases, LC.get_load_cases())

    with open("VNDiagram/critical_cases.txt", "w") as file:
        file.write(str(critical_cases))
    