import math
from variables import *
from Extrasheet import *


def stallSpeedWingLoading(CL_max, approach_velocity, mass_fraction, density = rhostd): # OK
    ratio_approach_to_stall_speed = 1.23  # Defined in CS-25
    return (1/mass_fraction) * (density * CL_max * (approach_velocity / ratio_approach_to_stall_speed )**2) / 2

def LandingFieldLength(Clmax, Lfd, mfrac,  Clfl = 0.45, density=1.225): # OK
    return ((1/mfrac)*(Lfd/Clfl)*((density*Clmax)/2))


def cruiseTTW(mass_fraction, C_D0, mach, aspect_ratio, span_efficiency_factor, wing_loading, cruise_altitude, std_temp = 288.15, thetaTbreak = 1.6): # OK
    # ISA
    a = -0.0065
    T_h = std_temp-0.0065*cruise_altitude
    p_h = 101325*(T_h/std_temp)**(-9.81/(a*287))
    density = p_h/(287*T_h)
    velocity = mach * (math.sqrt(1.4*287*T_h))

    # delta/theta T computation
    deltaT = (p_h*(1+((1.4-1)/2)*mach**2)**(1.4/(1.4-1)))/101325
    thetaT = (T_h*(1+(mach**2)*((1.4-1)/2)))/288.15
    
    if thetaT < thetaTbreak:
        thrust_lapse = deltaT*(1-(0.43+0.014*5)*math.sqrt(mach))
    else:
        thrust_lapse = deltaT*(1-(0.43+0.014*5)*math.sqrt(mach) - 3*(thetaT*thetaTbreak)/(1.5+mach))
    
    cruise = float(((mass_fraction*wing_loading)/(math.pi*0.5*density*(velocity**2)*span_efficiency_factor*aspect_ratio)) + 
                   (mass_fraction / thrust_lapse) * (((C_D0 * 0.5 * density * velocity**2) / (mass_fraction * wing_loading))))
    return cruise


def climbGradientTTW(mass_fraction, climb_gradient, wing_loading,  C_D0, aspect_ratio, span_efficiency_factor, OEI=False, NE = 2):
    thrust_lapse = Alpha(C_D0, aspect_ratio, wing_loading, span_efficiency_factor, mass_fraction, 0)
    
    TTW = (mass_fraction / thrust_lapse) * (climb_gradient + 2 * math.sqrt(C_D0/(math.pi * aspect_ratio * span_efficiency_factor)))
    
    if OEI:
        return TTW * (NE/ (NE-1))
    return TTW

def takeOffFieldLength(wing_loading, TO_field_length, rho,  AR, span_efficiency_factor, h2, k_t = 0.85, g = 9.80665, OEI = False, NE = 2):
    alphaT = Alpha(CD_0_phases[3], AR, wing_loading, oswald_phases[3], 1)
    if OEI:
        TTW = (1.15 * alphaT * math.sqrt(wing_loading/(k_t* g * rho * math.pi * AR * span_efficiency_factor))) + 4 * h2/TO_field_length
    else:
        TTW = (1.15 * alphaT * math.sqrt((NE/(NE-1))*wing_loading/(k_t* g * rho * math.pi * AR * span_efficiency_factor))) + 4 * h2/TO_field_length * (NE/(NE-1))
    return TTW

def climbRateTTW(climb_rate_requirement, mass_fraction, C_D0, aspect_ratio, wing_loading, span_efficiency_factor, density):
    thrust_lapse = Alpha(C_D0, aspect_ratio, wing_loading, span_efficiency_factor, mass_fraction,  climb_altitude=7400)
    climb_altitude = 7400
    std_temp = 288.15
    a = -0.0065
    T_h = std_temp-0.0065*climb_altitude
    p_h = 101325*(T_h/std_temp)**(-9.81/(a*287))
    
    density = p_h/(287*T_h)
    return (mass_fraction / thrust_lapse) * ((math.sqrt(((climb_rate_requirement**2)/(mass_fraction * wing_loading))*((density * math.sqrt(C_D0 * math.pi * aspect_ratio * span_efficiency_factor))/2)))+(2*(math.sqrt(C_D0/(math.pi * aspect_ratio * span_efficiency_factor)))))
