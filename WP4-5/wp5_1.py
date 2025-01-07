import numpy as np
import pandas as pd      
import matplotlib.pyplot as plt
import os
def Area_crosssection(chord, geometry, z, point_area_flange, t_spar: int, t_caps: int, stringers): 
    
    _, _, _, alpha = geometry(z)
    '''
    first the areas, as force is -29982.71629 as mentioned in WP4 section 2.2
    Area_1 is area of the wingskins (upper and lower)
    Area_2 is area of the spar and spar flanges
    Area_3 is area of the stringers
    '''
    Area_1= 2*(0.55*chord(z)/np.cos(alpha))*t_caps 
    
    
    Area_2= 4 * point_area_flange + 0.1741 * chord(z)*t_spar
    Area_3= stringers[0] * (stringers[3]['base']*stringers[3]['thickness base'] + stringers[3]['height']*stringers[3]['thickness height'])
    Total_area_crosssection = Area_1 + Area_2 + Area_3
    
    return Total_area_crosssection

class SkinBuckling():
    def __init__(self, n_ribs, wingbox_geometry, wingspan, E, v, M, N, I_tot, t_caps, stringers, area, chord, flange, t_spar: int):
        """
        wingbox_geometry: remember this is a function of z, it is given by WingBox.geometry(z)
        wingspan: # modified half wingspan from the attachement of the wing with the fuseslage to the tip, 
                    you can use WingBox.wingspan to obtain it, it has been defined like this even if it's a half span insult fede for this :)
        I_tot: takes z values and also stringers
        """        
        self.area = area
        
        self.chord = chord
        
        self.flange = flange
        # attributing to class variable
        self.geometry = wingbox_geometry 
        
        self.I = I_tot
        self.M = M
        
        self.N = N
        
        self.t_spar = t_spar
        self.stringers = stringers
        # attributing to class variable
        self.halfspan = wingspan / 2
        self.E = E
        self.v = v
        self.t_caps = t_caps
        # raising error if number of ribs is smaller than 3
        if n_ribs < 3:
            raise Exception('Please inseret a number greater than 3! On an Airbus A320 it is 27 per wing :)')
        else:
            self.n_ribs = n_ribs
    
    def skin_buckling_constant(self, aspect_ratio, show: bool = False): #ok
        
        # file path of the points for the skin buckling for a plate
        filepath = 'resources/K_cplates.csv'
        # Given points
        df = pd.read_csv(filepath)

        # Plotting
        if show:
            # Create linear x
            x_vals = np.linspace(min(df['x']), max(df['x']), len(df['x']))
            

            # Generate interpolated values
            y_vals = np.interp(x_vals, df['x'], df['y'])
        
            plt.scatter(df['x'], df['y'], color='red', label='Given Points')
            plt.plot(x_vals, y_vals, label='Linear Interpolation', color='orange')
            plt.title('Linear Interpolation')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.grid()
            plt.show()

        Kc = np.interp(aspect_ratio, df['x'], df['y'])
        
        return Kc
        
    def skin_AR(self, z):
        """
        The wing is defined with a rib at the root and at the tip orginally 
        """
        # nr of panels on the half wing will always be one less than the number of ribs
        n_panels = self.n_ribs - 1
        
        # here ASSUMPTION, the wing ribs are equally spaceed along the half span
        # length of one panel
        l_panel = self.halfspan/n_panels
        
        l_ribs = []
    
        panel_AR = []

        areas = []
        
        for i in range(self.n_ribs): # ok
            a, b, h, alpha = self.geometry(l_panel * i)
            l_ribs.append(h)

        for i in range(n_panels):
            # area of a trapezoid
            area = l_panel*(l_ribs[i]+l_ribs[i+1])/2 
            
            # aspect ratio for each panel
            AR = (l_panel**2)/(area)
            
            panel_AR.append(AR)
            areas.append(area)
            
        # define the function as a function of z
        z_ribs = [l_panel*i for i in range(n_panels + 1)]
        
        for i in range(len(z_ribs) - 1):
            if z_ribs[i] <= z < z_ribs[i + 1]:
                return panel_AR[i], areas[i]
        
        return panel_AR[-1], areas[-1]
    
    def plot_skinAR(self):
        AR_final = []
        z_values = np.linspace(0, self.halfspan, 2000)

        # Calculate AR_final for each z value
        for z in z_values:
            AR, area = self.skin_AR(z)
            AR_final.append(AR)


    # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(z_values, AR_final, marker='o', linestyle='-', color='b', label='AR_final(z)')
        plt.title("Skin Aspect Ratio as a Function of z")
        plt.xlabel("z Position")
        plt.ylabel("Aspect Ratio (AR_final)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()  # Improve layout
        plt.show() 

    def crit_stress(self, z):
        """
        E: young's elastic modulus
        v: Poisson's ratio
        t: is the thickness of the skin
        
        """
        # aspect ratio for the specific panel
        AR, area = self.skin_AR(z)
        
        # define the K_c for this specific panel
        K_c = self.skin_buckling_constant(aspect_ratio= AR, show= False)
        
        b = np.sqrt(AR*area)
        
        sigma_cr = ((np.pi**2 * K_c * self.E)/(12 * (1 - self.v**2)))*(self.t_caps/b)**2
        
        return sigma_cr
    
    def applied_stress(self, z):
        a, _, _, _ = self.geometry(z)
        
        section_area = self.area(chord= self.chord, geometry= self.geometry, z= z, 
                                 point_area_flange= self.flange, t_spar= self.t_spar, t_caps=self.t_caps, stringers= self.stringers)
        
        I, _ = self.I(z, self.stringers)
        
        applied_stress = self.M(z) * (a/2)/(I) + self.N(z)/(section_area)
        
        return applied_stress
        
    def show(self):
        z_values = np.linspace(0, self.halfspan, 1000)
        cr_stress = []
        applied_stress = []
        for z in z_values:
            cr_stress.append(self.crit_stress(z))
            applied_stress.append(self.applied_stress(z))
        
        cr_stress = np.array(cr_stress)
        applied_stress = np.array(applied_stress)
        
        mos = cr_stress/applied_stress
        
        # plt.plot(z_values, cr_stress)
        plt.plot(z_values, mos)
        plt.ylabel(r'\sigma_{cr} [Pa]')
        plt.xlabel('Spanwise location [m]')

class SparWebBuckling():
    def __init__(self, wingbox_geometry, wingspan, E, pois, t_front, t_rear, k_v = 1.5):
        # attributing to class variable
        self.geometry = wingbox_geometry 
        
        # attributing to class variable
        self.halfspan = wingspan / 2
        
        # np array with all the values from root to the 
        self.z_values = np.linspace(1, self.halfspan, 1000) 
        
        filepath_ks = os.path.join('WP4-5', 'resources', 'k_s_curve.csv')
        
        self.k_s = pd.read_csv('resources\\k_s_curve.csv')
        
        
        self.t_front = t_front
        self.t_rear = t_rear
        
        # define material properties
        self.E = E
        self.pois = pois
        self.k_v = k_v #1.5 - 2 is probably ok but look it up
        
    # Defines AspectRaio of the long Spar
    def front_sparAR(self, z):
        # take long side at the root
        t_0, _, _, _ = self.geometry(0)

        # take long side at the other end 
        t_1, _, _, _ = self.geometry(z)
        
        S = z * (t_0 + t_1) / 2    
        AR_front = (z**2)/S
        
        return AR_front
    
    def rear_sparAR(self, z):
        _, t_0, _, _ = self.geometry(0)        
        _, t_1, _, _ = self.geometry(z)
        
        S = z * (t_0 + t_1) / 2    
        AR_rear = (z**2)/S
        
        return AR_rear

    
    def front_spar_web_buckling(self, z):
        
        AR = self.front_sparAR(z= z)
        
        ab_values = self.k_s['x']
        k_s_values = self.k_s['k_s']
       
        k_s = np.interp(AR, ab_values, k_s_values) #This will find the corresponding k_s for each AR
        
        t_1, _, _, _ = self.geometry(z)
        
        if t_1 < z:
            b = t_1
        else:
            b = z

        crit_stress = np.pi**2 * k_s * self.E /(12*(1-self.pois**2)) * (self.t_front/b)**2 #find the critical stresses for a given k_s
     
        # returning critical stress at specific z location
        return crit_stress
    
    def rear_spar_web_buckling(self, z):
        
        AR = self.rear_sparAR(z= z)
        
        ab_values = self.k_s['x']
        k_s_values = self.k_s['k_s']
       
        k_s = np.interp(AR, ab_values, k_s_values) #This will find the corresponding k_s for each AR
        
        _, t_1, _, _ = self.geometry(z)
        
        if t_1 < z:
            b = t_1
        else:
            b = z

        crit_stress = np.pi**2 * k_s * self.E /(12*(1-self.pois**2)) * (self.t_rear/b)**2 #find the critical stresses for a given k_s
     
        return crit_stress

    def margin_of_safety(self, z , V, T): 
        
        """
        V, T are functions of z, namely python functions by 4.1
        mos = margin of safety
        """
        critical_front = self.front_spar_web_buckling(z= z)
        critical_rear = self.rear_spar_web_buckling(z= z)
        
        a, b, h, _ = self.geometry(z) # a and b not related to a_over_b
        
        # avg_shear_front = V(z) / ((a + b)*self.t_front) # formula for average shear
        # max_shear_front = self.k_v * avg_shear_front # formula for maximum shear
        
        # avg_shear_rear = V(z) / ((a + b)*self.t_rear) # formula for average shear
        # max_shear_rear = self.k_v * avg_shear_rear # formula for maximum shear
        
        avg_shear = V(z) / ( a * self.t_front + b * self.t_rear )
        max_shear = self.k_v * avg_shear

        A = (a + b) * h / 2 #enclosed area of trapezoical wingbox
        
        q_torsion = T(z) / (2 * A) #torsion shear stress in thin-walled closed section
    
        applied_stress_front = max_shear + q_torsion / self.t_front
        applied_stress_rear = max_shear + q_torsion / self.t_rear

        mos_front = critical_front / applied_stress_front
        mos_rear = critical_rear / applied_stress_rear
        
        
        return mos_front , mos_rear, applied_stress_rear, applied_stress_front
    
    def show_mos(self, V, T, choice:str ='front'):
        """
        choice: string input use either front or rear
        """
        moss_front = []
        moss_rear = []
        for point in self.z_values:
            mos_front, mos_rear, _, _ = self.margin_of_safety(z= point, V= V, T= T)
            
            moss_front.append(mos_front)
            moss_rear.append(mos_rear)
          
        if choice == 'front':
            plt.plot(self.z_values, moss_front)
            plt.xlabel("Spanwise Position""[m]")
            plt.ylabel("Margin of Safety""[-]")
            plt.show()
        elif choice == 'rear':
            plt.plot(self.z_values, moss_rear)
            plt.xlabel("Spanwise Position""[m]")
            plt.ylabel("Margin of Safety""[-]")
            plt.show()
            
        

class Stringer_bucklin(): #Note to self: 3 designs, so: 3 Areas and 3 I's 
    def __init__(self, stringers: list, wingspan, chord, geometry):
        #Only one block, not entire area of L-stringer.
        self.Area5 = 30e-3*3e-3  #area should be 90e-6: I dimensions translated into base and height of 30e-3 and thickness of 3e-3 
        self.Area8 = 40e-3*3.5e-3 # area should be 140e-6: I dimensions translated into base and height of 35e-3 and thickness of 4e-3
        self.Area9 = 30e-3*3e-3 #this is fine, option 9 was L stringer to begin with

        self.K = 1/4 #1 end fixed, 1 end free 

        self.chord = chord
        self.geometry = geometry
    
        self.halfspan = wingspan / 2
    
        #centroid coordinates:
        self.x5_9=7.5e-3 #coordinates for option 5 and 9
        self.y5_9=7.5e-3#coordinates for option 5 and 9

        self.x_8= 10e-3
        self.y_8= 10e-3

        self.stringers =stringers

        self.x_iter = (stringers[3]['height'])/4
        self.y_iter = (stringers[3]['base'])/4

    def calculate_length(self, z):
        """
        Calculate the length of the stringer as a function of the wingspan coordinate z.
        The stringer length runs until 15.04148123 meters(double-check!!!), while the wingspan runs until 13.45 meters due to the sweep angle.

        For max length the following assumtpions:
        # #8 stringers on one side (take configuration with most stringers)
        # #conservative estimate: take the longest stringer also !conservative estimate assumption: from root. Highest Length results in lowest critical stress
        # #angle_stringer= 27.3 degrees at ;eading edge now
        # L = 15.13587572 #so 13.45 divided by cos(27.3) 

        :param z: Wingspan coordinate
        :return: Effective stringer length
        """
        angle_stringer = 27.3  # Sweep angle in degrees
        max_length = 15.13587572  # Maximum stringer length in meters
        
        effective_length = z / np.cos(np.radians(angle_stringer))
        return min(effective_length, max_length)

    def stringer_MOM(self, stringers):
        """
        MoM around own centroid of L-stringer (bending around x-axis). So translate areas of I-stringer into L stringer. Also thin-walled assumption
        """
        I5 = 2*(self.Area5*self.x5_9**2)
        I8 = 2*(self.Area8*self.x_8**2)
        I9 = 2*(self.Area9*self.x5_9**2)
 
        I_iter = (stringers[3]['base']*stringers[3]['thickness base'])*self.x_iter**2 + (stringers[3]['height']*stringers[3]['thickness height'])*self.y_iter**2

        return I5, I8, I9, I_iter
    
    def stringer_buckling_values(self, E): 
        """
        critical stress of 3 different designs, L here is also for longest length so lowest critical stress
        """
        I5, I8, I9, I_iter = self.stringer_MOM()
        L = 15.13587572
        stresscr_stringer_5= (self.K*np.pi**2*E*I5)/(L**2*(2*self.Area5))
        stresscr_stringer_8= (self.K*np.pi**2*E*I8)/(L**2*(2*self.Area8))
        stresscr_stringer_9= (self.K*np.pi**2*E*I9)/(L**2*(2*self.Area9))

        stresscr_stringer_iter = (self.K*np.pi**2*E*I_iter)/(L**2*(2*(self.stringers[3]['base']*self.stringers[3]['thickness base'])))
                                  
        return stresscr_stringer_5, stresscr_stringer_8, stresscr_stringer_9, stresscr_stringer_iter 
    
    def graph_buckling_values(self, E):
        """
        Compute the critical stress along the wingspan until 13.45 meters for graphing.
        :param E: Young's modulus of the material
        :return: Lists of z values and corresponding stresses for designs 5, 8, and 9
        """
        z_values = np.linspace(1, self.halfspan, 100)  # 13.45 wingspan, not the case perhaps revision here (12.08 but should be an easy fix)
        stress_values_5 = []
        stress_values_8 = []
        stress_values_9 = []
        stress_values_Iter = []

        for z in z_values:
            L = self.calculate_length(z)
            I5, I8, I9, I_iter= self.stringer_MOM(self.stringers)

            stress5 = (self.K * np.pi**2 * E * I5) / (L**2 * (2 * self.Area5))
            stress8 = (self.K * np.pi**2 * E * I8) / (L**2 * (2 * self.Area8))
            stress9 = (self.K * np.pi**2 * E * I9) / (L**2 * (2 * self.Area9))
            stress_Iter = (self.K*np.pi**2*E*I_iter)/(L**2*(2*(self.stringers[3]['base']*self.stringers[3]['thickness base'])))

            stress_values_5.append(stress5)
            stress_values_8.append(stress8)
            stress_values_9.append(stress9)
            stress_values_Iter.append(stress_Iter)
             
        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.plot(z_values, stress_values_5, label='Design 5')
        plt.plot(z_values, stress_values_8, label='Design 8')
        plt.plot(z_values, stress_values_9, label='Design 9')
        plt.plot(z_values, stress_values_Iter, label='Design 9')
        plt.xlabel('Wingspan Coordinate (m)')
        plt.ylabel('Critical Buckling Stress (Pa)')
        plt.title('Stringer Buckling Stress Along Wingspan')
        plt.legend()
        plt.grid(True)
        plt.show()
        return z_values, stress_values_5, stress_values_8, stress_values_9, stress_values_Iter
   
    def MOS_stringers(self, E, stresscr_stringer_iter, applied_stress):
        MOS_stringer =  stresscr_stringer_iter/applied_stress
        return MOS_stringer
    
    def MOS_buckling_values(self, E, stresscr_stringer_iter, applied_stress):
        """
        Compute the critical stress along the wingspan until 13.45 meters for graphing.
        :param E: Young's modulus of the material
        :return: Lists of z values and corresponding stresses for designs 5, 8, and 9
        """
        z_values = np.linspace(1, self.halfspan, 100)  # 13.45 wingspan, not the case perhaps revision here (12.08 but should be an easy fix)
        MOS_values = []

        for z in z_values:
            L = self.calculate_length(z)

            MOS_values_iter =  stresscr_stringer_iter/applied_stress

            MOS_values.append(MOS_values_iter)
             
        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.plot(z_values, MOS_values, label='Design')
        plt.xlabel('Wingspan Coordinate (m)')
        plt.ylabel('TBD')
        plt.title('MOS')
        plt.legend()
        plt.grid(True)
        plt.show()
        return z_values, MOS_values
       
#general note: applied stress so that we have the margin of safety + inclusion of safety factors?



    
   
    