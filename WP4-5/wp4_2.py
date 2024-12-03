from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import math as math


# authors: Federico, Ben, Anita, Winston

#general: assumption is symmetric wing box utilised
class WingBox():
    def __init__(self, t: int, c_r: int, c_t: int, wingspan: int, intersection: int, tr:int = None):
        """
        c_r is chord root at the half of the fuselage
        stringers: list [number of stringers, percentage of span until they continue, type, dimensions(in a further list) dict type], must be an integer for the code to work
        dimensions: dict type changes in base of the used stringer
        L type stringer: {'base':, 'height':, 'thickness base':, 'thickness height':} [m]
        I type stringer: {'base':, 'top':, 'web height':, 'thickness top':, 'thickness web':, 'thickness base':} [m]
        distance: tuple, distance from centroid (x, y)
        intersection is the percentage of the wingspan where the wing cuts the fuselage
        
        """
        self.c_t = tr * c_r if c_t is None else c_t # tip chord [m]        
                             
        self.c_r = c_r - c_r*(1-(self.c_t /c_r))*intersection # redefining chord root
        
        self.t = t   # wingbox thickness, constant thickness in the cross sectiona nd along z assumed [m]
        
        self.deflections = pd.DataFrame(columns = ['Load [Nm]', 'z location [m]', 'Displacement [m]',
                                                    'Rotation [rad]', 'Moment of Inertia I [m^4]', 'Polar moment of Inertia J [m^4 (??)]'])
        self.wingspan = wingspan
        
        self.tiplocation = self.wingspan/2 - (self.wingspan/2) * intersection
             
    def chord(self, z): #TESTED OK
        # returns the chord at any position z in meters, not a percentage of halfspan, on 28/11 it can go from 0 to b/2 - intersection*b/2
        c = self.c_r - self.c_r * (1 - (self.c_t / self.c_r)) * (z / ((self.wingspan / 2)))
        return c

    def geometry(self, z: int):
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
      
    def bending (self, z, M, E):
        I = self.MOM_total()
        v_double_dot = lambda z: M/(-E*I)
        
        vdot = integrate.quad(v_double_dot, 0, z)
        v = integrate.quad(vdot, 0, z)
        
        if v > 0.15 * self.wingspan:
            print("Max Tip Displacement Exceeded", "Displacement =", v )
        else:
            print("Max Tip Displacement OK", "Displacement =", v )
        return v
    
    def centroid(self, z, stringers): # c-chord, t-thickness, alpha
        a, b, h, alpha = self.geometry(z)
        x_stringers, y_stringers, area_stringer, stringers_span = self.stringer_geometric(self, z, stringers)
        A = [b*self.t, a*self.t, h/np.cos(alpha)*self.t, h/np.cos(alpha)*self.t] #Areas of the components [longer side, shorter side, oblique top, oblique bottom]
        X = [0, h, 0.5*h/np.cos(alpha), 0.5*h/np.cos(alpha)]                     # X positions of the components
        Y = [0, 0, -0.5*a+0.5*h/np.sin(alpha), +0.5*a-0.5*h/np.sin(alpha)]       # Y positions of the components


        for i in range(len(x_stringers)):
            A.append(area_stringer[i])
            X.append(x_stringers[i])
            Y.append(y_stringers[i])

        for i in range(len(x)):
            weights_X = A[i]*X[i]
            weights_Y = A[i]*Y[i]
        
        x = weights_X/sum(A) #x position of the centroid
        y = weights_Y/sum(A)  #y position of the centroid
        
        return x, y

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
    
    def MOM_total_plots(self, z):
      

    def polar (self, z, t): # T : torsion, 
        a, b, h, alpha = self.geometry(z)
        A = h * (a + b) / 2               # Area of cross section [m^2]
        S = a + b + 2 * (h/np.cos(alpha)) # Perimetre of cross section [m]  
        #r = ((x_pos_string - x)**2 + (y_pos_string - y)**2)**0.5 # Distance from a stringer to centroid
        # stein = Area_string * (r**2)
        J = ((4*t*A**2)/S)
        return J
    
    def Jplots(self, z):
        ts = [0.001, 0.002, 0.003, 0.004, 0.005]
        z = np.linspace(0, self.tiplocation)
        
        for t in range(len(ts)):
            plt.plot(z, self.polar(z, ts[t]))        
        
        plt.grid(True)
        plt.legend()
        plt.show()
        return plt.gcf()
    
    def torsion (self, z, J, T: int, G, x_pos_string,y_pos_string, x, y, Area_string ): # T : torsion, 
       
        thetadot = lambda z: (T) / (J * G)

        theta = integrate.quad(thetadot, 0, z)
        if theta > np.deg2rad(abs(10)):
            print("Wing Tip Max. Rotation Exceeded", "Displacement =", np.rad2deg(theta))
        else:
            print("Wing Tip Max. Rotation Allowed", "Displacement =", np.rad2deg(theta))
        return theta

    def show(self, load, modulus, choice: str): 
        """
        load: int function representing the internal load of the wing [N]
        modulus: either E or G depending on the analysis [N/m2]
        halfspan: halfspan length [m]
        choice: 'bending' or 'torsion', string input
        """
        type = ['bending', 'torsion']
        
        z = np.linspace(0, self.tiplocation) # range of z values to show the plot
        
        self.deflections['Load [Nm]'] = 0
        self.deflections['z location [m]'] = z
        
        self.deflections['Moment of Inertia I [m^4]'] = 0
        self.deflections['Polar moment of Inertia J [m^4 (??)]'] = 0
            
        if choice == type[0]: 
            # bending diagram is chosen
            
            temp_v = []
            
            for i in z:
                # iterating through each value of z and evaluating the displacement at that point
                v = self.bending(z = i, M = load, E = modulus)
                
                temp_v.append(v)
            
            self.deflections['Displacement [m]'] = temp_v
            self.deflections['Rotation [rad]'] = np.zeros(len(z))
            
        elif choice == type[1]: 
            # Torque diagram is chosen
            
            temp_theta = []
            
            for i in z:
                # iterating through each value of z and evaluating the rotation at that point
                theta = self.torsion(z = i, T = load, G = modulus)
                
                temp_theta.append(v)
            
            
            self.deflections['Displacement [m]'] = np.zeros(len(z))
            self.deflections['Rotation [rad]'] = temp_theta

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
        
        for i in range(1, stringer_per_side+1):
            x_stringers.append(i * h / (np.cos(alpha) * (stringer_per_side + 1)))
            x_stringers.append(i * h / (np.cos(alpha) * (stringer_per_side + 1)))
            y_stringers.append(- a / 2 + (i * h / (np.sin(alpha) * (stringer_per_side + 1))))
            y_stringers.append(a / 2 -(i * h / (np.sin(alpha) * (stringer_per_side + 1))))
            
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
