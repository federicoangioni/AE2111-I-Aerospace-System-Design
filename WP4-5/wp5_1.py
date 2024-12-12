from scipy.interpolate import interp1d
import numpy as np
import pandas as pd



class SkinBuckling():
    def __init__(self):
        pass
        
        
    def skin_buckling_constant(self, aspect_ratio):
        
        # file path of the points for the skin buckling for a plate
        filepath = 'WP4-5/resources/K_cplates.csv'
        # Given points
        df = pd.read_csv(filepath)
        
        # Create linear interpolating function
        linear_interp = interp1d(df['x'], df['y'], kind='next')

        # Generate interpolated values
        x_vals = np.linspace(min(df['x']), max(df['x']), len(df['x']))
        y_vals = linear_interp(df['x'])

        # Plotting
        import matplotlib.pyplot as plt
        plt.scatter(df['x'], df['y'], color='red', label='Given Points')
        plt.plot(x_vals, y_vals, label='Linear Interpolation', color='orange')
        plt.title('Linear Interpolation')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid()
        plt.show()

        #Kc = interp_function(aspect_ratio)
        #print(f"Interpolated value at AR={x_query}: kc={y_query}")
        

    # def SkinAspectRatio(number_of_ribs, wing_span, geometry):
    #     number_of_panels = number_of_ribs - 1
    #     length_of_the_panel = wing_span/number_of_panels

    #     for (int i =0, i<=number_of_ribs)

    #     a, b, h, alpha = geometry(length_of_the_panel)

    #     length_of_ribs = 


skin = SkinBuckling()
skin.skin_buckling_constant(8)