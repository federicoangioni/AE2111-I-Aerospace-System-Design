from scipy.interpolate import interp1d
import numpy as np
from variables import *



class SkinBuckling():
    def __init__():
        
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

    def SkinAspectRatio(number_of_ribs, wing_span, geometry, z):
        half_wing =wing_span/2
        number_of_panels = number_of_ribs - 1
        length_of_the_panel = half_wing/number_of_panels
        length_of_ribs = []
        panel_area = []
        panel_AR = []

        for i in range(number_of_ribs + 1):
            a, b, h, alpha = geometry(length_of_the_panel * i)
            length_of_ribs.append(h)

        for i in range(number_of_ribs -1):
            panel_area.append(0.5*length_of_the_panel*(length_of_ribs[i]+length_of_ribs[i+1]))
            AR = panel_area[i]/(length_of_the_panel **2)

            if AR >=1:
                panel_AR.append(AR)
            else:
                panel_AR.append(panel_area[i]/length_of_ribs[i]**2)

        for i in range(number_of_panels-1):
            if z => length_of_the_panel * i and z<length_of_the_panel * (i+1):
                AR_final = panel_AR[i]
            else: AR_final = -100000
        
        return AR_final
    

    def StressSkinBuckling(panel_AR, E, Poisson, constant, t, z):
        stress = np.pi**2 * constant*E/(12*(1-Poisson**2))*(t/b)**2
        return stress




class RibWebBuckling():
    print("hello world")