from typing import Union
import numpy as np
import math

# Speeds
def speed_from_lift(Lift, rho, S, Cl):
    return math.sqrt((2 * Lift) / (rho * S * Cl))

def dive_speed(speed_cruise, factor = 1.25):
    return speed_cruise * factor

def limiting_speeds_upper_clean(weight, n_max, rho, S, Cl_max, speed_cruise):
    minimum_speed = 0
    speed_stall_max_n = math.sqrt(n_max) * speed_from_lift(weight, rho, S, Cl_max)
    speed_dive = dive_speed(speed_cruise)
    return minimum_speed, speed_stall_max_n, speed_dive

def limiting_speeds_lower_clean(weight, n_min, rho, S, Cl_max, speed_cruise):
    minimum_speed = 0
    speed_stall_max_n = - math.sqrt(abs(n_min)) * speed_from_lift(weight, rho, S, Cl_max)
    speed_dive = dive_speed(speed_cruise)
    return minimum_speed, speed_stall_max_n, speed_cruise, speed_dive

def Limiting_speeds_upper_flaps(weight, n_max_flaps, rho, S, Cl_max_flaps, Cl_max_clean, speed_cruise):
    minimum_speed = 0
    speed_stall_max_n_flap = math.sqrt(n_max_flaps) * speed_from_lift(weight, rho, S, Cl_max_flaps)
    max_speed = math.sqrt(n_max_flaps) * speed_from_lift(weight, rho, S, Cl_max_clean)
    return minimum_speed, speed_stall_max_n_flap, max_speed


# Load Factors as function of Speed
def n_max(weight_kg):
    kg_to_lbs = 2.20462
    n_max_from = None
    return 2.1 + (2400 / ((kg_to_lbs * weight_kg) + 1000))

def n_positive_stall_speed(speed, stall_speed_no_flaps, n_max):
    return None

if __name__ == "__main__":
    print("running wp4-3.py")
