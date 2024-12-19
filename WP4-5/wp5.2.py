import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from main import g_moment
from variables import b

def x_func(z):
    return 0.5 * (1 - z / 10) 

def y_func(z):
    return 0.3 * z

def Ixx_func(z):
    return 100 + 5 * z 

def Iyy_func(z):
    return 150 + 3 * z

def Ixy_func(z):
    return 0 * z


def stress_z(Mx_func, My_func, Ixx_func, Iyy_func, Ixy_func, z, x, y):

    # Evaluate the moment and inertia functions at spanwise location z
    Mx = Mx_func(z)
    My = My_func(z)
    Ixx = Ixx_func(z)
    Iyy = Iyy_func(z)
    Ixy = Ixy_func(z)

    # Calculate numerator and denominator
    numerator = (Mx * Iyy - My * Ixy) * y + (My * Ixx - Mx * Ixy) * x
    denominator = Ixx * Iyy - Ixy**2

    # division by zero?
    if denominator == 0:
        raise ValueError("Denominator is zero. Check inertia values.")

    # Calculate normal stress
    sigma_z = numerator / denominator
    return sigma_z

def calculate_stress_distribution(z_points, x_func, y_func, Mx_func, My_func, Ixx_func, Iyy_func, Ixy_func):
    """
    Calculate the normal stress distribution along the wing span.
    """
    stress_distribution = []
    for z in z_points:
        x = x_func(z)
        y = y_func(z)
        sigma_z = stress_z(Mx_func, My_func, Ixx_func, Iyy_func, Ixy_func, z, x, y)
        stress_distribution.append(sigma_z)
    return np.array(stress_distribution)

def main():
    # Define spanwise locations
    z_points = np.linspace(0, b, 100)  # Wing span from 0 to 10 meters

    # Calculate stress distribution
    stress_distribution = calculate_stress_distribution(
        z_points=z_points,
        x_func=x_func,
        y_func=y_func,
        Mx_func=g_moment,
        My_func= 0,  # Assuming same moment for simplicity
        Ixx_func=Ixx_func,
        Iyy_func=Iyy_func,
        Ixy_func=Ixy_func
    )

    # Find critical stress
    max_stress = max(stress_distribution)
    critical_z = z_points[np.argmax(stress_distribution)]
    print(f"Maximum Normal Stress: {max_stress:.2f} Pa at Spanwise Location: {critical_z:.2f} m")

    # Plot stress distribution
    plt.figure(figsize=(12, 6))
    plt.plot(z_points, stress_distribution, label="Normal Stress Distribution", color="blue")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Spanwise Location (z) [m]", fontsize=12)
    plt.ylabel("Normal Stress [Pa]", fontsize=12)
    plt.title("Normal Stress Distribution Along the Wing Span", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()

    