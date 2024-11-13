import numpy as np
"""
24.86  span
4.27 root chord
0.3115 taper ratio

"""

# This calculates the corresponding Swf to varying flap outer-edge position
def swf(c_r, tr, a = 0.7):
    
    b = 4.27 - 4.27*(1-0.3115)*a
    swf= a*24.86*((4.27+b)/2)
    print("swf = ", swf, "C_oe = ", b)
    return(a)

swf(0.7)

# Theotretical swf based on Lift increase
def theoryswf(a):
    b = 0.8*a
    d = np.cos(hinge(0.6))
    r = b/(0.9*1.612*d)
    print("Swf/S = ", r)
    return(a)

# hinge angle: varies with spar placement
def hinge(a):
    b = np.tan(0.468) - a * ((2*4.33)/24.75)*(1-0.32)
    c = np.arctan(b)
    print("Hinge Angle = ", c, "rad")
    return(c)