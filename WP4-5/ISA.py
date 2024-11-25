import math
from variables import R, gamma

def speed_of_sound(T):
    return math.sqrt(gamma * R * T)

def temperature(altitude):
    if altitude < 11000:
        return 288.15 - 0.0065 * altitude
    else:
        return 216.65
    
def pressure(altitude):
    if altitude < 11000:
        return 101325 * (temperature(altitude) / 288.15) ** (9.81 / (0.0065 * R))
    else:
        return 22632 * math.exp(-9.81 * (altitude - 11000) / (R * 216.65))

def density(altitude):
    return pressure(altitude) / (R * temperature(altitude))

class AtmosphericConditions():
    """
    Altitude in meters
    """
    def __init__(self, altitude) -> None:
        self.altitiude = altitude
        self.temperature = temperature(altitude)
        self.pressure = pressure(altitude)
        self.density = density(altitude)
        self.speed_of_sound = speed_of_sound(self.temperature)

    def get_temperature(self):
        return self.temperature
    
    def get_pressure(self):
        return self.pressure
    
    def get_density(self):
        return self.density
    
    def get_speed_of_sound(self):
        return self.speed_of_sound
        