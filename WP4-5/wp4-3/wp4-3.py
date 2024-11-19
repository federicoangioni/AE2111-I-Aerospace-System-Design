import numpy as np
import math

def speed_from_lift(Lift, rho, S, Cl):
    return math.sqrt((2 * Lift) / (rho * S * Cl))

def n_max(weight_kg):
    kg_to_lbs = 2.20462
    return 2.1 + (2400 / ((kg_to_lbs * weight_kg) + 1000))

if __name__ == "__main__":
    print("running wp4-3.py")