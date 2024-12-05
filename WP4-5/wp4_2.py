from scipy import integrate
from scipy.integrate import cumtrapz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import math as math
from scipy.interpolate import interp1d


# authors: Federico, Ben, Anita, Winston

#general: assumption is symmetric wing box utilised
class WingBox():
    def __init__(self, c_r: int, c_t: int, wingspan: int, intersection: int,  t_spar: int, t_caps: int, tr:int = None,):
        """
        c_r is chord root at the half of the fuselage
        stringers: list [number of stringers, percentage of span until they continue, type, dimensions(in a further list) dict type], must be an integer for the code to work
        dimensions: dict type changes in base of the used stringer
        L type stringer: {'base':, 'height':, 'thickness base':, 'thickness height':} [m]
        I type stringer: {'base':, 'top':, 'web height':, 'thickness top':, 'thickness web':, 'thickness base':} [m]
        distance: tuple, distance from centroid (x, y)
        intersection is the percentage of the wingspan where the wing cuts the fuselage
        wingspan, PUT ORIGINAL WINGSPAN WITH ORIGIN AT THE MIDDLE OF THE PLANE, as of 03/12 it is at 26.9 m
        """
        self.c_t = tr * c_r if c_t is None else c_t # tip chord [m]        
                             
        self.c_r = c_r - c_r*(1-(self.c_t /c_r))*intersection # redefining chord root
                
        self.deflections = pd.DataFrame()
        self.wingspan = (wingspan- intersection * wingspan)
        
        self.t1, self.t2 = t_spar, t_caps
        
        self.z = np.linspace(0, self.wingspan/2, 100) # will be useful to iterate through the functions
        
        print(f"Wing span modified goes from 0 to {np.round(self.wingspan/2, 3)}")
            
    def chord(self, z): # ok
        # returns the chord at any position z in meters, not a percentage of halfspan, on 28/11 it can go from 0 to b/2 - intersection*b/2
        c = self.c_r - self.c_r * (1 - (self.c_t / self.c_r)) * (z / ((self.wingspan / 2)))
        return c

    def geometry(self, z: int): # ok
        # returns the wing box geometry (side lenghts) at any position z
        a = 0.1013 * self.chord(z)        # trapezoid longer  [m]
        b = 0.0728 * self.chord(z)        # trapezoid shorter [m]
        h = 0.55 * self.chord(z)          # trapezoid height  [m]
        alpha = np.arctan(((a-b)/2)/h)    # angle angle [rad]
        return a, b, h, alpha
    
    def show_geometry(self, z):
        a, b, h, alpha = self.geometry(z)
        
        plt.plot([0, 0], [b/2, -b/2])
        plt.plot([0, h], [b/2, a/2])
        plt.plot([h, h], [a/2, -a/2])
        plt.plot([0, h], [-b/2, -a/2])
        
        plt.show()
        plt.clf()    
 
    def Spar(self, Spar_thickenss, multiplication_factor, z):
        a, b, h, alpha = self.geometry(z)
        Point_area_flange = Spar_thickenss**2 * multiplication_factor
        Spar_thickness = Spar_thickenss
        Flange_spar_position_x = [0,h/2,h/2,0]
        Flange_spar_position_y = [a/2, b/2, -b/2, -a/2]

        return Flange_spar_position_x, Flange_spar_position_y ,Spar_thickness, Point_area_flange        
      
    def bending (self, z, M, E):
        I = self.MOM_total()
        v_double_dot = lambda z: M/(-E*I)
        
        vdot = integrate.quad(v_double_dot, 0, z)
        v = integrate.quad(vdot, 0, z)
        
        if v > 0.15 * self.wingspan:
            print("Max Tip Displacement Exceeded", "Displacement =", v, (v/self.wingspan)*100, "(% Wingspan)" )
        else:
            print("Max Tip Displacement OK", "Displacement =", v , (v/self.wingspan)*100, "(% Wingspan)")
        return v
    
    def centroid(self, z, stringers): # c-chord, t-thickness, alpha
        a, b, h, alpha = self.geometry(z)
        # x_stringers, y_stringers, area_stringer, stringers_span = self.stringer_geometric(self, z, stringers)
        # Flange_spar_position_x, Flange_spar_position_y ,Spar_thickness, Point_area_flange = Spar(self, Spar_thickenss, multiplication_factor, z)
        # A = [b*self.t, a*self.t, h/np.cos(alpha)*self.t, h/np.cos(alpha)*self.t] #Areas of the components [longer side, shorter side, oblique top, oblique bottom]
        # X = [0, h, 0.5*h/np.cos(alpha), 0.5*h/np.cos(alpha)]                     # X positions of the components
        # Y = [0, 0, -0.5*a+0.5*h/np.sin(alpha), +0.5*a-0.5*h/np.sin(alpha)]       # Y positions of the components


        # for i in range(len(x_stringers)):
        #     A.append(area_stringer[i])
        #     X.append(x_stringers[i])
        #     Y.append(y_stringers[i])
        
        # for i in range(len(Flange_spar_position_x)):
        #     A.append(Point_area_flange)
        #     X.append(Flange_spar_position_x[i])
        #     Y.append(Flange_spar_position_y[i])


        # for i in range(len(x)):
        #     weights_X = A[i]*X[i]
        #     weights_Y = A[i]*Y[i]
        
        # x = weights_X/sum(A) #x position of the centroid
        # y = weights_Y/sum(A)  #y position of the centroid
        
        # return x, y
    
    def plot_centroid(self, z_range, stringers):
        x_vals = []
        y_vals = []
    
        for z in z_range:
            x, y = self.centroid(z, stringers)
            x_vals.append(x)
            y_vals.append(y)
    
        plt.figure(figsize=(10, 6))
        plt.plot(z_range, x_vals, label="Centroid X-Position")
        plt.plot(z_range, y_vals, label="Centroid Y-Position")
        plt.xlabel("z (Position along beam)")
        plt.ylabel("Centroid Position")
        plt.title("Centroid Positions as a Function of z")
        plt.legend()
        plt.grid(True)
        plt.show()
   

    def MOMEWB (self, z, x, y): #Moment of inertia for empty wing box, #ci and cj are related to distance from centroid/coordinate system
        a, b, h, alpha = self.geometry(z)
        # old version: alpha = np.arctan(((a-b)/2)/h)
        
        ci1= h - x
        ci2= x
        ci3= np.cos(alpha)*(((h/np.cos(alpha)))/2) - x
        cj3= (b/2)+ np.sin(alpha)*((h/np.cos(alpha))/2)

        #Split into 3 section: 1 is the short vertical bar, 2 is the long vertical bar, and 3 are the bars at an angle
        #section 1:
        I1xx = (self.t*b**3)/12 + 0 #I know it's silly
        I1yy = 0 + (self.t*b)*ci1**2

        #section 2:
        I2xx = (self.t*a**3)/12 + 0
        I2yy = 0 + (self.t*a)*ci2**2

        #section 3 (so both bars): #bar at angle practically same as bar: 0.5501845713 chord
        I3xx= (2/12)*self.t*(np.sin(alpha)**2)*((h/np.cos(alpha))**3) +2*((h/np.cos(alpha))*self.t)*cj3**2
        I3yy= (2/12)*self.t*(np.cos(alpha)**2)*((h/np.cos(alpha))**3) +2*((h/np.cos(alpha))*self.t)*ci3**2

        #Total moments of inertia of wing box:
        I_wingbox_xx = I1xx+I2xx+I3xx
        I_wingbox_yy = I1yy+I2yy+I3yy

        return I_wingbox_xx, I_wingbox_yy
        
    def MOM_total (self, I_wingbox_xx, I_wingbox_yy, I_xx_stringers_steiner, I_yy_stringers_steiner): #total Moment of Intertia (so empty wing box and stringers)

        I_total_xx = I_wingbox_xx + I_xx_stringers_steiner
        I_total_yy = I_wingbox_yy + I_yy_stringers_steiner

        return (I_total_xx, I_total_yy)

    def polar (self, z): # T : torsion, 
        a, b, h, alpha = self.geometry(z)
        A = h * (a + b) / 2               # Area of cross section [m^2]
        denom = (b/self.t1) + 2*((h/np.cos(alpha))/self.t2) + (a/self.t1) #t1 is spar thickness, t2 is thickness of horizontal portion

        J = (4*A**2)/denom
        return J
    
    def Jplots(self, z):
        t1 = [0.001, 0.002, 0.003, 0.004, 0.005]
        t2 = [0.001, 0.002, 0.003, 0.004, 0.005]
        z = np.linspace(0, self.tiplocation)
        for i in range(len(t1)): 
            for j in range(len(t2)): 
                plt.plot(z, self.polar(z, t1[i], t2[j]), label=f'Thickness {t1[i]}, {t2[j]}')
        #for t in range(len(t1), len(t2)):
            #plt.plot(z, self.polar(z, t1[t], t2[t]), label = f'Thickness: {t1 = t1[t], t2 = t2[2]}')        
        plt.grid(True)
        plt.legend(title = "Thickness(t)", loc = "upper right")
        plt.xlabel('z [m]')
        plt.ylabel("Torsional Stiffness [mm$^4$]")
        plt.title("Torsional Stiffness for varying Spanwise Locations and Thickness")
        plt.show()
        return plt.gcf()
    
    def torsion (self, z, T: int, G): # ok
        """
        Takes as input a z numpy array which goes from 0 to the half span
        """
        # T is defined with z fro 0 to b/2 in m
        thetadot = lambda x: (T(x)) / (self.polar(x) * G)
        
        # interpolating the orginal function to avoid discontinuities along the integration
        thetad = interp1d(z, thetadot(z), kind='cubic', fill_value="extrapolate")
        
        thetas = []

        for i in range(len(z)):
            theta, error = integrate.quad(thetad, 0, z[i])
            
            thetas.append(theta)

        return thetas

    def show(self, z, loads, moduli, limits: int, plot: bool = False, save: bool= False, degrees= False): 
        """
        load: int function representing the internal load of the wing [N]
        moduli: [E, G] in Pa or N/m2
        limit: it is the maximum allowable displacement [mm, deg] or [bending, torsion]
        halfspan: halfspan length [m] 
        choice: 'bending' or 'torsion', string input
        degrees: plots the torsion diagram in degrees instead of radians
        """
        # changes on 04/12 we won't plot one displacement at a time but all together
        torque = loads[2]
        moment = loads[1]
        
        self.deflections['z location [m]'] = z
        
        self.deflections['Moment of Inertia I [mm^4]'] = 0 #?
        self.deflections['Polar moment of Inertia J [mm^4]'] = self.polar(z= z)
            
        vs = self.bending(z = z, M = moment, E = moduli[0])
            
        self.deflections['Displacement [m]'] = vs
        
        # Torque diagram 
        
        thetas = self.torsion(z = z, T = torque, G = moduli[1])            
        
        self.deflections['Rotation [rad]'] = thetas
        self.deflections['Rotation [deg]'] = np.degrees(thetas)
        
        if (self.deflections['Rotation [rad]'] > np.radians(limits[1])).any().any():
            print("Wing Tip Max. Rotation Exceeded", "Max displacement =", np.degrees(max(self.deflections['Rotation [rad]'])))
        else:
            print("Wing Tip Max. Rotation Allowed", "Max displacement =", np.degrees(max(self.deflections['Rotation [rad]'])))
        # plotting
        if plot and degrees:
            # divide in subplots @todo
            plt.plot(self.deflections['z location [m]'], np.degrees(self.deflections['Rotation [rad]']))
            plt.axhline(y = limits[1], color = 'r', linestyle = '-', lw= 1, dashes=[2, 2])
            plt.xlabel("Span wise position [m]")
            plt.ylabel(r"$\theta$ rotation [rad]")
            plt.grid()
            plt.show()
                      
        # write a CSV with all the information        
        with open('WP4-5/deflections.csv',  'w', encoding = 'utf=8') as file:
            self.deflections.to_csv(file)


    def stringer_geometry(self, z, stringers):
        a, b, h, alpha = self.geometry(z)
        
        # defining the variables for the stringers
        if type(stringers) != list or len(stringers) == 3:
            raise Exception("Insert the value in a list of length 2")
        if stringers[0] % 2 != 0: 
            raise Exception("Please Insert an even number")
        else:
            stringer_per_side = int(stringers[0]/2)
            stringers_span = stringers[1]
            stringers_type = stringers[2]
            dimensions = stringers[3]
        
        if stringers_type == "L":
            area_stringer = dimensions["base"]*dimensions["height"] + dimensions["thickness base"]*dimensions["thickness height"]
        
        elif stringers_type == "I":
            area_stringer = dimensions["base"]*dimensions["thickness base"] + dimensions["web"]*dimensions["thickness web"] + dimensions["top"]* dimensions["thickness top"]
        
        x_stringers = []
        y_stringers = []

        num_stringers = stringers[0]
        
        stringer_spacing = (h/np.cos(alpha)) / (num_stringers + 1)  # Spacing between stringers on the bars

        for i in range(1, stringer_per_side+1): # Upperside
               # Distance along the inclined bar
               distance_along_bar = i * stringer_spacing

               # Position of the stringer in the global coordinate system (origin is at the long vertical bar)
            #    x_pos_string = h - (distance_along_bar * np.cos(alpha))   
               x_stringers.append(h - (distance_along_bar * np.cos(alpha)))      
               y_stringers.append((b/2) + (distance_along_bar * np.sin(alpha)))

        for i in range(1, stringer_spacing+1): #lowerside
            distance_along_bar = i * stringer_spacing
            x_stringers.append(h - (distance_along_bar * np.cos(alpha)))
            y_stringers.append(-((b/2) + (distance_along_bar * np.sin(alpha))))  
            
        return x_stringers, y_stringers, area_stringer, stringers_span

    def stringer_I(self, z, stringers): # check this again, too many functions in one
        a, b, h, alpha = self.geometry(self, z)
        x_stringers, y_stringers, area_stringer, stringers_span = self.stringer_geometric(z, stringers)
        x, y = self.centroid(z, stringers)
        num_stringers = stringers[0]
        
        stringer_spacing = (h/np.cos(alpha)) / (num_stringers + 1)  # Spacing between stringers on the bars

        # For each bar, calculate contributions
        I_xx_stringers_steiner, I_yy_stringers_steiner = 0, 0
        for i in range(1, num_stringers+1):

               # Distance along the inclined bar
               distance_along_bar = i * stringer_spacing

               # Position of the stringer in the global coordinate system (origin is at the long vertical bar)
               x_pos_string = h - (distance_along_bar * np.cos(alpha))         
               y_pos_string = (b/2) + (distance_along_bar * np.sin(alpha))
               """
               Note: position is now on the bars, we may need to adjust this a little based on the assumption
               """
               # Contribution to moments of inertia using parallel axis theorem
               I_xx_sub = area_stringer * (y_pos_string - y) ** 2
               I_yy_sub = area_stringer * (x_pos_string - x) ** 2

               # Add contributions to total
               I_xx_stringers_steiner += I_xx_sub
               I_yy_stringers_steiner += I_yy_sub

         # Double the total contributions because we have two bars
        I_xx_stringers_steiner *= 2
        I_yy_stringers_steiner *= 2
        
        return I_xx_stringers_steiner, I_yy_stringers_steiner, x_pos_string, y_pos_string
