from scipy.interpolate import interp1d
import numpy as np


def SkinBucklingConstant(aspect_ratio):

    # Given points
    x_points = [0.9, 1, 1.5, 2, 3, 4, 5]
    y_points = [14, 10.25, 8.75, 8, 7.5, 7.5, 7.5]

    # Create linear interpolating function
    linear_interp = interp1d(x_points, y_points, kind='linear')

    # Generate interpolated values
    x_vals = np.linspace(min(x_points), max(x_points), 500)
    y_vals = linear_interp(x_vals)

    # Plotting
    import matplotlib.pyplot as plt
    plt.scatter(x_points, y_points, color='red', label='Given Points')
    plt.plot(x_vals, y_vals, label='Linear Interpolation', color='orange')
    plt.title('Linear Interpolation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

    Kc = interp_function(aspect_ratio)
    print(f"Interpolated value at AR={x_query}: kc={y_query}")
    return Kc