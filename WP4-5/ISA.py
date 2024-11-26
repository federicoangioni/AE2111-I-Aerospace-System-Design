import math
from variables import R, gamma

def speed_of_sound(T: float) -> float:
    return math.sqrt(gamma * R * T)

def temperature(altitude: float) -> float:
    if altitude < 11000:
        return 288.15 - 0.0065 * altitude
    else:
        return 216.65
    
def pressure(altitude: float) -> float:
    if altitude < 11000:
        return 101325 * (temperature(altitude) / 288.15) ** (9.81 / (0.0065 * R))
    else:
        return 22632 * math.exp(-9.81 * (altitude - 11000) / (R * 216.65))

def density(altitude):
    return pressure(altitude) / (R * temperature(altitude))

class AtmosphericConditions():
    """
    This class gives the atmospheric conditions at a given altitude

    Methods:
    --------
    get_temperature() -> float:
        Returns the temperature at the given altitude
    get_pressure() -> float:
        Returns the pressure at the given altitude
    get_density() -> float:
        Returns the density at the given altitude
    get_speed_of_sound() -> float:
        Returns the speed of sound at the given altitude
    """
    def __init__(self, altitude: float) -> None:
        """
        Constructs all atributtes of the atmospheric conditions

        Parameters:
        -----------
        altitude: float
            Altitude in meterss
        """
        self.altitiude = altitude
        self.temperature = temperature(altitude)
        self.pressure = pressure(altitude)
        self.density = density(altitude)
        self.speed_of_sound = speed_of_sound(self.temperature)

    def get_temperature(self) -> float:
        """Return the Temperature for the given atmosphere"""
        return self.temperature
    
    def get_pressure(self) -> float:
        """Return the Pressure for the given atmosphere"""
        return self.pressure
    
    def get_density(self) -> float:
        """Return the Density for the given atmosphere"""
        return self.density
    
    def get_speed_of_sound(self) -> float:
        """Return the Speed of sound for the given atmosphere"""
        return self.speed_of_sound
        