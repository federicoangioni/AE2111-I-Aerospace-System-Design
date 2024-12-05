import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from wp4_2 import WingBox
from variables import *


x_data = np.array([1, 2, 3, 4, 5])
y1_data = np.array([1, 4, 9, 16, 25])   # y = x^2

# Create the first interpolated function
f1_interp = interp1d(x_data, y1_data, kind='linear')

# Define the second function explicitly in terms of x
def f2(x):
    return x + 1  # Example: simple linear function

# Create the division function
def division_function(x):
    return f1_interp(x) / f2(x)


wingbox = WingBox(c_r = c_r, c_t = None, wingspan= b, area_factor_flanges=12, t_spar=0.001, t_caps=0.001, intersection= (d/2) / (b/2), tr = tr)

stringers = [20, 0.9, 'L', {'base': 10e-3, 'height': 5e-3, 'thickness base': 2e-3, 'thickness height': 2e-3}]
# print(wingbox.stringer_geometry(0, stringers= stringers))

print(wingbox.centroid(0, stringers))

# z = np.linspace(0, 1)
wingbox.show_geometry(0, stringers)
# wingbox.Jplots(z)

# plot_centroid(self, z_range, stringers)