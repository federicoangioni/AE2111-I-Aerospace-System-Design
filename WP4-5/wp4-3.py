import numpy as np
import math

def speed_from_lift(Lift, rho, S, Cl):
    return math.sqrt((2 * Lift) / (rho * S * Cl))


if __name__ == "__main__":
    print("running wp4-3.py")