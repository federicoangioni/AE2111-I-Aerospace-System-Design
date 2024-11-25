from matplotlib import pyplot as plt
import numpy as np
import math
import ISA
import variables as var


# Speeds
def speed_from_lift(Lift, rho, S, Cl):
    return math.sqrt((2 * Lift) / (rho * S * Cl))

def dive_speed_formula(speed_cruise, factor = 1.25):
    return speed_cruise * factor


# Load Factors as function of Speed
# Line B

# Line A, G, F
def n_stall_speed(speed, stall_speed):
    return (speed / stall_speed) ** 2

# Line D
def n_linear_lower_part(speed, cruise_speed, dive_speed):
    return (1 / (dive_speed - cruise_speed)) * (speed - dive_speed)


# Lines
# upper clean
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

def V_n_line_upper_flaps(speeds, n_max_flaps, speed_stall_flaps, speed_stall_clean):
    list = [
        n_stall_speed(speed, speed_stall_flaps) if n_stall_speed(speed, speed_stall_flaps) <= n_max_flaps 
        else n_max_flaps  
        for speed in speeds 
        if speed <= speed_stall_clean * math.sqrt(n_max_flaps)
    ]
    return np.array(list)

    
class V_n_diagram():
    def __init__(self, weight_kg, density, temperature, CL_max_clean, CL_max_flapped, wing_area = 71.57, cruise_mach = 0.77) -> None:
        self.weight_kg = weight_kg
        self.weight = weight_kg * var.g
        self.density = density
        self.temperature = temperature
        self.speed_of_sound = ISA.speed_of_sound(var.gamma, var.R, temperature)
        self.CL_max_clean = CL_max_clean
        self.CL_max_flapped = CL_max_flapped
        self.n_min = -1
        self.n_max_flapped = 2
        self.wing_area = wing_area
        self.V_cr = cruise_mach * self.speed_of_sound

    def generate_points(self, number_of_points = 1000):
        max_n = self.n_max_formula()
        speed_stall_clean = speed_from_lift(self.weight, self.density, self.wing_area, self.CL_max_clean)
        speed_stall_flaps = speed_from_lift(self.weight, self.density, self.wing_area, self.CL_max_flapped)

        speeds = np.linspace(0, dive_speed_formula(self.V_cr), number_of_points)
        n_upper_clean = V_n_line_upper_clean(speeds, max_n, speed_stall_clean, self.V_cr)
        n_lower_clean = V_n_line_lower_clean(speeds, self.n_min, speed_stall_clean, self.V_cr)
        n_upper_flaps = V_n_line_upper_flaps(speeds, self.n_max_flapped, speed_stall_flaps, speed_stall_clean)

        speeds = np.append(speeds, dive_speed_formula(self.V_cr))
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
        plt.xlabel("Speed [m/s]")
        plt.ylabel("Load factor")
        plt.title("V-n diagram")
        plt.show()

    def n_max_formula(self):
        weight_kg = self.weight_kg
        kg_to_lbs = 2.20462
        n_max_formula = 2.1 + (2400 / ((kg_to_lbs * weight_kg) + 1000))
        return min(2.5,max(n_max_formula, 3.8))



if __name__ == "__main__":
    print("running wp4-3.py")
    weight_kg = 19593  #kg


    Vn1 = V_n_diagram(weight_kg, var.rho0, var.T0, 1.41, 2.55)
    Vn1.generate_points()
    Vn1.show()


        