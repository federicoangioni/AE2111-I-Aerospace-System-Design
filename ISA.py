import math

g_0 = 9.80665
R = 287.05
a = -0.0065
T_0 = 288.15
P_0 = 101325
rho_0 = 1.225

def Temperature(height):
    return T_0 + a * height

def Pressure(height):
    return P_0 * math.pow((Temperature(height) / T_0), -(g_0 / (a * R)))

def Density(height):
    return Pressure(height) / (R * Temperature(height))