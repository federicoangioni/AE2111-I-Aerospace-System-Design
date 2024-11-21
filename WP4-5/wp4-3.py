from matplotlib import pyplot as plt
import numpy as np
import math
# from ..\variables import *
import ISA


# Speeds
def speed_from_lift(Lift, rho, S, Cl):
    return math.sqrt((2 * Lift) / (rho * S * Cl))

def dive_speed(speed_cruise, factor = 1.25):
    return speed_cruise * factor

# Points A & B
def limiting_speeds_upper_clean(weight, n_max, rho, S, Cl_max, speed_cruise):
    minimum_speed = 0
    speed_stall_max_n = math.sqrt(n_max) * speed_from_lift(weight, rho, S, Cl_max)
    speed_dive = dive_speed(speed_cruise)
    return minimum_speed, speed_stall_max_n, speed_dive

# Points E & D & C
def limiting_speeds_lower_clean(weight, rho, S, Cl_max, speed_cruise, n_min=-1):
    minimum_speed = 0
    speed_stall_max_n = - math.sqrt(abs(n_min)) * speed_from_lift(weight, rho, S, Cl_max)
    speed_dive = dive_speed(speed_cruise)
    return minimum_speed, speed_stall_max_n, speed_cruise, speed_dive

# Points F & G
def Limiting_speeds_upper_flaps(weight, n_max_flaps, rho, S, Cl_max_flaps, Cl_max_clean):
    minimum_speed = 0
    speed_stall_max_n_flap = math.sqrt(n_max_flaps) * speed_from_lift(weight, rho, S, Cl_max_flaps)
    max_speed = math.sqrt(n_max_flaps) * speed_from_lift(weight, rho, S, Cl_max_clean)
    return minimum_speed, speed_stall_max_n_flap, max_speed


# Load Factors as function of Speed
# Line B
def n_max(weight_kg):
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
def V_n_line_upper_clean():
    pass


def plotting(weight, rho, S, Cl_max_clean, Cl_max_flaps, speed_cruise):

    stall_speed_clean_n1 = speed_from_lift(weight, rho, S, Cl_max_clean)

    return stall_speed_clean_n1

if __name__ == "__main__":
    print("running wp4-3.py")

    # Test variables
    g = 9.81 # m/s^2
    weight_kg = 19593 # kg
    weight_N = weight_kg * g # N
    rho = 1.225 # kg/m^3
    S = 71.57 # m^2
    C_L_max_land = 2.55
    C_L_max_clean = 1.41
    M_cr = 0.77
    gamma = 1.4
    R = 287.05
    T = 288.15
    speed_of_sound = ISA.speed_of_sound(gamma, R, T)
    V_cr = M_cr * speed_of_sound

    stall_speed_clean_n1 = plotting(weight_N, rho, S, C_L_max_clean, C_L_max_land, V_cr)
    print(stall_speed_clean_n1)


        