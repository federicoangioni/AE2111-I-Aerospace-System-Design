import math
from variables import *

def Alpha(Zlcd, AR, WingLoading, Oswald, mass_fraction, climb_altitude = 0, std_temp = 288.15, density = 1.225): # climbtemp = 240.05,
    # ISA
    a = -0.0065
    T_h = std_temp-0.0065*climb_altitude
    p_h = 101325*(T_h/std_temp)**(-9.81/(a*287))
    


    thetaTbreak = 1.7 # 1.6 < thetaTbreak < 1.8
    
    ClimbRateCl = math.sqrt(Zlcd*math.pi*AR*Oswald)
  
    speed = math.sqrt(mass_fraction*WingLoading*(2/density)*(1/ClimbRateCl))
    mach = speed/math.sqrt(1.4*287*T_h)
    pt = p_h * (1+(mach**2)*((1.4-1)/2))**(1.4/(1.4-1))
    Tt = T_h*(1+(mach**2)*((1.4-1)/2))
    deltat = pt / 101325 
    thetat = Tt / 288.15
    if thetat <= thetaTbreak:
        alphaT = deltat*(1-(0.43+0.014*5)*math.sqrt(mach))
    else:
        alphaT = deltat*(1-(0.43+0.014*5)*math.sqrt(mach) - 3*(thetat*thetaTbreak)/(1.5+mach))
    
    return alphaT



    

