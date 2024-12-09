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
    def __init__(self, c_r: int, c_t: int, wingspan: int, intersection: int,  area_factor_flanges :int, t_spar: int, t_caps: int, tr:int = None):
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
        self.wingspan_og = wingspan

        self.wingspan = (wingspan- intersection * wingspan)
        
        self.t_spar, self.t_caps = t_spar, t_caps
        
        self.z = np.linspace(0, self.wingspan/2, 100) # will be useful to iterate through the functions
        
        self.flanges_area_ratio = area_factor_flanges # ratio of height-to-thickness of spar flanges
        
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
    
    def show_geometry(self, z, stringers): # kinda useless buut may be useful for nice graphs
        a, b, h, alpha = self.geometry(z)
        
        plt.plot([0, 0], [a/2, -a/2])
        plt.plot([0, h], [a/2, b/2])
        plt.plot([h, h], [b/2, -b/2])
        plt.plot([0, h], [-a/2, -b/2])
        centroid = self.centroid(z, stringers)
        plt.scatter(centroid[0], centroid[1])
        plt.show()
        plt.clf()    
        print(a, b, h, alpha)
 
    def spar_flanges(self, z):
        a, b, h, alpha = self.geometry(z)
        
        point_area_flange = self.t_spar**2 * self.flanges_area_ratio
        flange_spar_pos = [(0, a/2), (h, b/2), (h, -b/2), (0, a/2)] # going in counterclockwise from upper right
        #Parallel axis theorem:
        Ixx_sparflanges= 2* point_area_flange * (b/2)**2 + 2* point_area_flange * (a/2)**2

      
        return flange_spar_pos, point_area_flange, Ixx_sparflanges

    def bending(self, z, M, E, stringers):
        
        
        v_double_dot = lambda x: M(x) /(-E*self.MOM_total(z=x, stringers=stringers)[0])
        
        
        
        v_double_dot_g = interp1d(z, v_double_dot(z), kind='cubic', fill_value="extrapolate")
        vs = []
        vdot_list =[]
        Is = []
        for i in range(len(z)):
            
            vdot, _ = integrate.quad(v_double_dot_g, 0, z[i])
    
            vdot_list.append(vdot)
            Is.append(self.MOM_total(z=z[i], stringers=stringers)[0])

        vdot_g =  interp1d(z, vdot_list, kind='cubic', fill_value="extrapolate")
    
        for i in range(len(z)):
            
            v, _ = integrate.quad(vdot_g, 0, z[i])
    
            vs.append(v)
        
        return vs
    
    def centroid(self, z, stringers): # c-chord, t-thickness, alpha
        a, b, h, alpha = self.geometry(z)
        
        x_stringers, y_stringers, area_stringer, stringers_span = self.stringer_geometry(z, stringers)
        
        flange_spar_pos, point_area_flange, Ixx_sparflanges = self.spar_flanges(z= z)
        
        area_trapezoid = [b*self.t_spar, a*self.t_spar, h/np.cos(alpha)*self.t_caps, h/np.cos(alpha)*self.t_caps] # Areas of the components [longer side, shorter side, oblique top, oblique bottom]
        x_trapezoid = [h, 0, 0.5*h/np.cos(alpha), 0.5*h/np.cos(alpha)]                                            # X positions of the components
        
        sum_trap_x = 0
        sum_flanges_x = 0
        for i in range(len(area_trapezoid)):
            sum_trap_x += area_trapezoid[i]*x_trapezoid[i]
            sum_flanges_x += point_area_flange*flange_spar_pos[i][0] # select x pos from tuple

         #2/3 contribution of stringers to centroid coordinates:
        num_stringers = stringers[0]
        #Only x-coordinate is relevant: x-coordinate of stringers
        x_coordinate_stringer = h - (((h/np.cos(alpha)) /2) * np.cos(alpha))
        weight_x_coordinate_stringer = x_coordinate_stringer * num_stringers * area_stringer #assuming number of stringers is total amount of stringers
        
        #3/3 contribution of spar flanges to centroid coordinates:
        areas = (point_area_flange)*4 + sum(area_trapezoid) + (num_stringers * area_stringer)
        
        x= (sum_trap_x + sum_flanges_x + weight_x_coordinate_stringer) / areas
        y = 0 # always midway between the two caps, as it's symmetric along x - axis
        return x, y
    
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
   

    def MOMEWB (self, z, stringers): #Moment of inertia for empty wing box, #ci and cj are related to distance from centroid/coordinate system
        x, y = self.centroid(z, stringers=stringers)
        a, b, h, alpha = self.geometry(z)
        # old version: alpha = np.arctan(((a-b)/2)/h)
        
        ci1= h - x
        ci2= x
        ci3= np.cos(alpha)*(((h/np.cos(alpha)))/2) - x
        cj3= (b/2)+ np.sin(alpha)*((h/np.cos(alpha))/2)

        #Split into 3 section: 1 is the short vertical bar, 2 is the long vertical bar, and 3 are the bars at an angle
        #section 1:
        I1xx = (self.t_spar*b**3)/12 + 0 #I know it's silly
        I1yy = 0 + (self.t_spar*b)*ci1**2

        #section 2:
        I2xx = (self.t_spar*a**3)/12 + 0
        I2yy = 0 + (self.t_spar*a)*ci2**2

        #section 3 (so both bars): #bar at angle practically same as bar: 0.5501845713 chord
        I3xx= (2/12)*self.t_caps*(np.sin(alpha)**2)*((h/np.cos(alpha))**3) +2*((h/np.cos(alpha))*self.t_caps)*cj3**2
        I3yy= (2/12)*self.t_caps*(np.cos(alpha)**2)*((h/np.cos(alpha))**3) +2*((h/np.cos(alpha))*self.t_caps)*ci3**2

        #Total moments of inertia of wing box:
        I_wingbox_xx = I1xx+I2xx+I3xx
        I_wingbox_yy = I1yy+I2yy+I3yy

        return I_wingbox_xx, I_wingbox_yy
        
    def MOM_total (self, z, stringers): #total Moment of Intertia (so empty wing box and stringers)
        
        I_wingbox_xx, I_wingbox_yy = self.MOMEWB(z, stringers=stringers)
        I_xx_stringers_steiner, I_yy_stringers_steiner, x_pos_string, y_pos_string = self.stringer_I(z, stringers=stringers)
        flange_spar_pos, point_area_flange, Ixx_sparflanges = self.spar_flanges(z)

        I_total_xx = I_wingbox_xx + I_xx_stringers_steiner +Ixx_sparflanges
        I_total_yy = I_wingbox_yy + I_yy_stringers_steiner
        
        return I_total_xx, I_total_yy

    def polar (self, z): # T : torsion, 
        a, b, h, alpha = self.geometry(z)
        A = h * (a + b) / 2               # Area of cross section [m^2]
        denom = (b/self.t_spar) + 2*((h/np.cos(alpha))/self.t_caps) + (a/self.t_spar) #t1 is spar thickness, t2 is thickness of horizontal portion

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

    def show(self, z, loads, moduli, limits: int, stringers, plot: bool = False, save: bool= False, degrees= False, filename=None): 
        """
        load: int function representing the internal load of the wing [N]
        moduli: [E, G] in Pa or N/m2
        limit: it is the maximum allowable displacement [mm, deg]
        halfspan: halfspan length [m] 
        choice: 'bending' or 'torsion', string input
        degrees: plots the torsion diagram in degrees instead of radians
        """
        # changes on 04/12 we won't plot one displacement at a time but all together
        moment = loads[0]
        torque = loads[1]
        
        
        self.deflections['z location [m]'] = z
        
        # self.deflections['Moment of Inertia I [mm^4]'] = self.MOM_total(z=z, stringers=stringers)[0]
        self.deflections['Polar moment of Inertia J [mm^4]'] = self.polar(z= z)
            
        vs = self.bending(z = z, M = moment, E = moduli[0], stringers=stringers)
            
        self.deflections['Displacement [m]'] = vs
        
        # Torque diagram 
        
        thetas = self.torsion(z = z, T = torque, G = moduli[1])            
        
        self.deflections['Rotation [rad]'] = thetas
        self.deflections['Rotation [deg]'] = np.degrees(thetas)
        
        if (abs(self.deflections['Displacement [m]']) > 0.15*self.wingspan_og).any().any():
            print("Max Tip Displacement Exceeded", "Displacement =", max(abs(self.deflections['Displacement [m]'])), (max(abs(self.deflections['Displacement [m]']))/self.wingspan_og)*100, "(% Wingspan)" )
        else:
            print("Max Tip Displacement OK", "Displacement =", max(abs(self.deflections['Displacement [m]'])) , (max(abs(self.deflections['Displacement [m]']))/self.wingspan_og)*100, "(% Wingspan)")
            
            
        if (abs(self.deflections['Rotation [rad]']) > np.radians(limits[1])).any().any():
            print("Wing Tip Max. Rotation Exceeded", "Max displacement =", max(abs(self.deflections['Rotation [deg]'])))
        else:
            print("Wing Tip Max. Rotation Allowed", "Max displacement =", max(abs(self.deflections['Rotation [deg]'])))
        # plotting
        if plot and degrees:
            # divide in subplots @todo
            fig, axs = plt.subplots(1, 2, figsize=(8, 5))
            axs[0].plot(self.deflections['z location [m]'], self.deflections['Rotation [deg]'])
            axs[0].axhline(y = limits[1], color = 'r', linestyle = '-', lw= 1, dashes=[2, 2])
            axs[0].set_xlabel("Span wise position [m]")
            axs[0].set_ylabel(r"$\theta$ rotation [deg]")
            axs[0].set_title("Rotation due to torsion")
            axs[0].grid()
            
            axs[1].plot(self.deflections['z location [m]'], self.deflections['Displacement [m]'])
            axs[1].axhline(y = np.sign(self.deflections['Displacement [m]'].iloc[-1])*0.15*self.wingspan_og, color = 'r', linestyle = '-', lw= 1, dashes=[2, 2])
            
            axs[1].set_xlabel("Span wise position [m]")
            axs[1].set_ylabel("Displacement due to bending moment [m]")
            axs[1].set_title("Displacement due to bending")
            axs[1].grid()
            plt.tight_layout()
            
            if save:
                plt.savefig(filename)
            plt.show()
                      
        # write a CSV with all the information        
        with open('deflections.csv',  'w', encoding = 'utf=8') as file:
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
            area_stringer = dimensions["base"]*dimensions["thickness base"] + dimensions["web height"]*dimensions["thickness web"] + dimensions["top"]* dimensions["thickness top"]
        
        x_stringers = []
        y_stringers = []

        num_stringers = stringers[0]
        
        stringer_spacing = (h/np.cos(alpha)) / (stringer_per_side + 1)  # Spacing between stringers on the bars

        for i in range(1, stringer_per_side+1): # Upperside
               # Distance along the inclined bar
               distance_along_bar = i * stringer_spacing

               # Position of the stringer in the global coordinate system (origin is at the long vertical bar)
            #    x_pos_string = h - (distance_along_bar * np.cos(alpha))   
               x_stringers.append(h - (distance_along_bar * np.cos(alpha)))      
               y_stringers.append((b/2) + (distance_along_bar * np.sin(alpha)))

        for i in range(1, stringer_per_side+1): #lowerside
            distance_along_bar = i * stringer_spacing
            x_stringers.append(h - (distance_along_bar * np.cos(alpha)))
            y_stringers.append(-((b/2) + (distance_along_bar * np.sin(alpha))))  
            
        return x_stringers, y_stringers, area_stringer, stringers_span

    def stringer_I(self, z, stringers): # check this again, too many functions in one
        a, b, h, alpha = self.geometry(z)
        x_stringers, y_stringers, area_stringer, stringers_span = self.stringer_geometry(z, stringers)
        x, y = self.centroid(z, stringers)
        stringer_per_side = int(stringers[0]/2)
        
        stringer_spacing = (h/np.cos(alpha)) / (stringer_per_side + 1)  # Spacing between stringers on the bars

        # For each bar, calculate contributions
        I_xx_stringers_steiner, I_yy_stringers_steiner = 0, 0
        for i in range(1, stringer_per_side +1):

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