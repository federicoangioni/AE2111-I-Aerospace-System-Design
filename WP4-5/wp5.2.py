import numpy as np
from scipy import interpolate

def stress_z(Mx_func, My_func, Ixx_func, Iyy_func, Ixy_func, z, x, y):
    """
    Calculate normal stress at point (x, y) in the cross-section along the wing span (z).

    Parameters:
    Mx_func, My_func : callable
        Functions for bending moments about x and y axes as a function of spanwise location (z).
    Ixx_func, Iyy_func, Ixy_func : callable
        Functions for second moments of area and product of inertia as a function of spanwise location (z).
    z : float
        Spanwise location along the wing.
    x, y : float
        Coordinates of the point in the cross-section.

    Returns:
    sigma_z : float
        Normal stress at point (x, y) in the cross-section at spanwise location z.
    """
    # Evaluate the moment and inertia functions at spanwise location z
    Mx = Mx_func(z)
    My = My_func(z)
    Ixx = Ixx_func(z)
    Iyy = Iyy_func(z)
    Ixy = Ixy_func(z)

    # Calculate numerator and denominator
    numerator = (Mx * Iyy - My * Ixy) * y + (My * Ixx - Mx * Ixy) * x
    denominator = Ixx * Iyy - Ixy**2

    # Check for potential division by zero
    if denominator == 0:
        raise ValueError("Denominator is zero. Check inertia values.")

    # Calculate normal stress
    sigma_z = numerator / denominator
    return sigma_z


def stress_distribution(Mx_func, My_func, Ixx_func, Iyy_func, Ixy_func, z_values, x, y):
  
    sigma_z_distribution = []
    for z in z_values:
        sigma_z = stress_z(Mx_func, My_func, Ixx_func, Iyy_func, Ixy_func, z, x, y)
        sigma_z_distribution.append(sigma_z)
    return np.array(sigma_z_distribution)