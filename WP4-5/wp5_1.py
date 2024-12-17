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

        #everything under this part relates to stringer buckling:
""""
Note to self: 3 designs, so: 3 Areas and 3 I's 
""""
Area5 = 30e-3*3e-3 #Only one block, not entire area of L-stringer. area should be 90e-6: I dimensions translated into base and height of 30e-3 and thickness of 3e-3 
Area8 = 40e-3*3.5e-3 #Only one block, not entire area of L-stringer. area should be 140e-6: I dimensions translated into base and height of 35e-3 and thickness of 4e-3
Area9 = 30e-3*3e-3 #Only one block, not entire area of L-stringer. this is fine, option 9 was L stringer to begin with
K = 1/4 #1 end fixed, 1 end free 

#calculation of length: 
#8 stringers on one side (take configuration with most stringers)
#conservative estimate: take the longest stringer also !conservative estimate assumption: from root. Highest Length results in lowest critical stress
#angle_stringer= 26.59493069 degrees at 1/9 of chord
L = 15.04148123 #so 13.45 divided by cos(26.5949)
#doublecheck value

#centroid coordinates:
x5_9=7.5e-3 #coordinates for option 5 and 9
y5_9=7.5e-3#coordinates for option 5 and 9

x_8= 10e-3
y_8= 10e-3

def Stringer_MOM ():#MoM around own centroid of L-stringer (bending around x-axis). So translate areas of I-stringer into L stringer. Also thin-walled assumption
    I5 = 2*(Area5*x5_9**2)

    I8 = 2*(Area8*x_8**2)

    I9 = 2*(Area9*x5_9**2)

    return I5, I8, I9

def Stringer_buckling (E, K, L, I5,I8,I9): #critical stress of 3 different designs
    stresscr_stringer_5= (K*np.pi**2*E*I5)/(L**2*(2*Area5))

    stresscr_stringer_8= (K*np.pi**2*E*I8)/(L**2*(2*Area8))
    
    stresscr_stringer_9= (K*np.pi**2*E*I9)/(L**2*(2*Area9))

    return stresscr_stringer_5, stresscr_stringer_8, stresscr_stringer_9

        #everything above this part relates to stringer buckling
 
   
    