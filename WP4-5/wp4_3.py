from matplotlib import pyplot as plt
import numpy as np
import math
from ISA import AtmosphericConditions as ISA
import variables as var


def Prandtl_Glauert_correction(Mach, CL):
    return CL / math.sqrt(1 - Mach ** 2)     


# Speeds
def speed_from_lift(Lift, rho, S, Cl):
    return math.sqrt((2 * Lift) / (rho * S * Cl))

def dive_speed_formula(speed_cruise, factor = 1.25):
    return speed_cruise * factor

def design_flap_speed(MLW, density, wing_area, CL_max_flapped):
    # TODO: check for the other minimums
    return 1.8 * speed_from_lift(MLW, density, wing_area, CL_max_flapped)

def n_stall_speed(speed, stall_speed):
    return (speed / stall_speed) ** 2

def n_linear_lower_part(speed, cruise_speed, dive_speed):
    return (1 / (dive_speed - cruise_speed)) * (speed - dive_speed)

def approach_speed(speed_stall_clean):
    return 1.3 * speed_stall_clean


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
            MTO_kg: float,
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
        self.MTO_kg = MTO_kg
        self.MTO = MTO_kg * var.g
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
        speed_design_flap = design_flap_speed(self.MLW, density, wing_area, CL_max_flapped)
        self.approach_speed = approach_speed(speed_stall_clean)

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
        n_upper_flaps = np.append(n_upper_flaps, 0)

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
        weight_kg = self.MTO_kg
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
    
    def get_approach_speed(self):
        return self.approach_speed

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
        self.speed_approach = Vn.get_approach_speed()

        self.flaps_infliction_point = self.speed_stall_flap * math.sqrt(self.max_n_flaps)
        self.clean_upper_infliction_point = self.speed_stall_clean * math.sqrt(self.max_n_clean)
        self.clean_lower_infliction_point = self.speed_stall_clean * math.sqrt(abs(self.min_n))
        
        
        self.critical_load_cases = np.array([
            self.case_given_speed_given_n(self.speed_stall_clean, flaps="TO"), # stall speed n = 1
            self.case_given_speed_given_n(self.speed_approach), # approach speed n = 1
            self.case_given_speed_given_n(self.speed_dive), # dive speed n = 1
            self.case_given_speed(self.flaps_infliction_point, self.n_upper_flaps, flaps="TO"), # flap inflection point (n=2)
            self.case_last_of_line(self.n_upper_flaps, flaps="TO"), # flap design speed (n=2?)
            self.case_given_speed(self.speed_approach, self.n_upper_clean), # clean upper approach speed
            self.case_given_speed(self.clean_upper_infliction_point, self.n_upper_clean), # clean upper inflection point
            self.case_last_of_line(self.n_upper_clean), # clean upper dive speed
            self.case_given_speed_given_n(self.n_upper_clean), # clean upper end at n=0
            self.case_given_speed(self.speed_cruise, self.n_lower_clean), # Clean lower cruise speed
            self.case_given_speed(self.clean_lower_infliction_point, self.n_lower_clean) # Clean lower inflection point (n=-1)
        ])

    def case_given_speed_given_n(self, speed: float, n=1, flaps = "Clean"):
        speed = self.speeds[np.where(self.speeds >= speed)[0][0]]
        load_factor = n
        weight = self.Vn.get_weight()
        return {"load_factor": load_factor, "speed": speed, "weight": weight, "flaps": flaps}
    
    def case_given_speed(self, speed: float, line: np.ndarray, flaps = "Clean"):
        speed = self.speeds[np.where(self.speeds >= speed)[0][0]]
        load_factor = line[np.where(self.speeds >= speed)[0][0]]
        weight = self.Vn.get_weight()
        return {"load_factor": load_factor, "speed": speed, "weight": weight, "flaps": flaps}
    
    def case_last_of_line(self, line: np.ndarray, flaps = "Clean"):
        load_factor = line[-2]
        speed = self.speeds[len(line) - 2]
        weight = self.Vn.get_weight()
        return {"load_factor": load_factor, "speed": speed, "weight": weight, "flaps": flaps}
    
    def get_load_cases(self):
        return self.critical_load_cases


if __name__ == "__main__":
    print("running wp4-3.py")
    MTOW_kg = 19593  #kg
    MLW_kg = 0.886 * MTOW_kg
    CL_max_clean = 1.41
    CL_max_flapped = 2.55


    Vn1 = VelocityLoadFactorDiagram(MTOW_kg, MLW_kg, MTOW_kg, 0, CL_max_clean, CL_max_flapped)
    Vn1.show()

    LC1 = LoadCases(Vn1)
    print(LC1.get_load_cases())
    


