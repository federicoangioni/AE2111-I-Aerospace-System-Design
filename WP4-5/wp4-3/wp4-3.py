from matplotlib import pyplot as plt
import numpy as np
import math
import ISA
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import variables


# Speeds
def speed_from_lift(Lift, rho, S, Cl):
    return math.sqrt((2 * Lift) / (rho * S * Cl))

def dive_speed_formula(speed_cruise, factor = 1.25):
    return speed_cruise * factor


# Load Factors as function of Speed
# Line B
def n_max_formula(weight_kg):
    kg_to_lbs = 2.20462
    n_max_formula = 2.1 + (2400 / ((kg_to_lbs * weight_kg) + 1000))
    return min(2.5,max(n_max_formula, 3.8))

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


def plotting(speeds, n_upper_clean, n_lower_clean, n_upper_flaps):
    plt.plot(speeds, n_lower_clean)
    plt.plot(speeds, n_upper_clean)
    plt.plot(speeds[:len(n_upper_flaps)], n_upper_flaps)
    plt.grid()
    plt.xlabel("Speed [m/s]")
    plt.ylabel("Load factor")
    plt.title("V-n diagram")
    plt.show()

if __name__ == "__main__":
    print("running wp4-3.py")

    # test values
    number_of_points = 1000
    #atmosphere
    density = 1.225
    R = 287.05
    Temperature = 288.15
    gamma = 1.4
    speed_of_sound = ISA.speed_of_sound(gamma, R, Temperature)
    gravitaional_acceleration = 9.80665 #m/s^2
    #aircraft
    weight_kg = 19593  #kg
    weight = weight_kg * gravitaional_acceleration  #N
    M_cr = 0.77
    speed_cruise = M_cr * speed_of_sound
    S = 71.57  #m^2
    Cl_max_clean = 1.41
    Cl_max_flaps_land = 2.55

    # test outputs
    speed_stall_clean = speed_from_lift(weight, density, S, Cl_max_clean)
    speed_stall_flaps = speed_from_lift(weight, density, S, Cl_max_flaps_land)
    max_n = n_max_formula(weight_kg)

    speeds = np.linspace(0, dive_speed_formula(speed_cruise), number_of_points)
    n_upper_clean = V_n_line_upper_clean(speeds, max_n, speed_stall_clean, speed_cruise)
    n_lower_clean = V_n_line_lower_clean(speeds, -1, speed_stall_clean, speed_cruise)
    n_upper_flaps = V_n_line_upper_flaps(speeds, 2, speed_stall_flaps, speed_stall_clean)
    speeds = np.append(speeds, dive_speed_formula(speed_cruise))
    n_upper_clean = np.append(n_upper_clean, 0)
    n_lower_clean = np.append(n_lower_clean, 0)

    print(f"""\
    speed_stall_clean:  {speed_stall_clean}
    speed_stall_flaps:  {speed_stall_flaps}
    v_dive:             {dive_speed_formula(speed_cruise)} 
    n_max:              {max_n}
    speeds:             {n_upper_clean} 
    length n_upper:     {len(n_upper_clean)}
    length n_lower:     {len(n_lower_clean)}
    length speeds:      {len(speeds)}
    """)

    plotting(speeds, n_upper_clean, n_lower_clean, n_upper_flaps)

        