import numpy as np
import pandas as pd      
import matplotlib.pyplot as plt
    

class SkinBuckling():
    def __init__(self, n_ribs, wingbox_geometry, wingspan, E, v, t_skin):
        """
        wingbox_geometry: remember this is a function of z, it is given by WingBox.geometry(z)
        wingspan: # modified half wingspan from the attachement of the wing with the fuseslage to the tip, 
                    you can use WingBox.wingspan to obtain it, it has been defined like this even if it's a half span insult fede for this :)
        """        
        
        # attributing to class variable
        self.geometry = wingbox_geometry 
        
        # attributing to class variable
        self.halfspan = wingspan / 2
        self.E = E
        self.v = v
        self.t = t_skin
        # raising error if number of ribs is smaller than 3
        if n_ribs < 3:
            raise Exception('Please inseret a number greater than 3! On an Airbus A320 it is 27 per wing :)')
        else:
            self.n_ribs = n_ribs
    
    def skin_buckling_constant(self, aspect_ratio, show: bool = False): #ok
        
        # file path of the points for the skin buckling for a plate
        filepath = 'WP4-5/resources/K_cplates.csv'
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

    def sigma_crit(self, z):
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
        
        sigma_cr = ((np.pi**2 * K_c * self.E)/(12 * (1 - self.v**2)))*(self.t/b)**2
        
        return sigma_cr
    
    def plot_sigma_cr(self):
        
        z_values = np.linspace(0, self.halfspan, 1000)
        sigmas = []
        
        for z in z_values:
            sigmas.append(self.sigma_crit(z))
            
        plt.plot(z_values, sigmas)
        plt.show()
        
        


# 






















class SparWebBuckling():
#     def __init__(self, wingbox_geometry: function, wingspan):
#         # attributing to class variable
#         self.geometry = wingbox_geometry 
        
#         # attributing to class variable
#         self.wingspan = wingspan
#         pass  

#     def chord(self, z, c_r, c_t, wingspan): 
#         # returns the chord at any position z in meters, not a percentage of halfspan, on 28/11 it can go from 0 to b/2 - intersection*b/2
#         c = c_r - c_r * (1 - (c_t / c_r)) * (z / ((wingspan / 2)))
#         return c
    
#     # Defines AspectRaio of the long Spar
#     def LongSparWebAR(self, z):
#         wing_span = self.wingspan
#         a = self.geometry(z)
#         LongSparWebAR_z = []
#         z_values = np.linspace(0,wing_span/2, 100) 
#         for z in z_values:
#             t_0 = a(0)
#             t_1 = a(z_values)
#             S = z * (t_0 + t_1) / 2    
#             AR = (z**2)/S
#             LongSparWebAR_z.append(AR)

#         return LongSparWebAR_z
    
#     def ShortSparWebAR(self, z):
#         wing_span = self.wingspan
#         b = self.geometry(z)
#         ShortSparWebAR_z = []
#         z_values = np.linspace(0,wing_span/2, 100) 
#         for z in z_values:
#             t_0 = b(0)
#             t_1 = b(z_values)
#             S = z * (t_0 + t_1) / 2    
#             AR = (z**2)/S
#             ShortSparWebAR_z.append(AR)
        
#         return ShortSparWebAR_z

    
    def front_spar_web_buckling(self, AR, E, t_sparweb, b, v):
        from k_s_curve import k_s_array
        AR = self.LongSparWebAR()
        k_s_array_np = np.array(k_s_array)
        ab_values = k_s_array_np[:, 0]
        k_s_values = k_s_array_np[:, 1]
        k_s_z_front = []
        crit_stress_z_front = []
        
        for i in range(len(AR)):
            k_s = np.interp(AR, ab_values, k_s_values) #This will find the corresponding k_s for each AR
            crit_stress = np.pi**2 * k_s * E /(12*(1-v**2)) * (t_sparweb/b)**2 #find the critical stresses for a given k_s
            k_s_z_front.append(k_s)
            crit_stress_z_front.append(crit_stress) # Stores the relevant critical stress in a list
        
#         return crit_stress_z_front
    
    def rear_spar_web_buckling(self, AR, E, t_sparweb, b, v):
        from k_s_curve import k_s_array
        AR = self.SparWebARSparWebAR()
        k_s_array_np = np.array(k_s_array)
        ab_values = k_s_array_np[:, 0]
        k_s_values = k_s_array_np[:, 1]
        k_s_z_back = []
        crit_stress_z_back = []
        for i in AR:
            k_s = np.interp(AR, ab_values, k_s_values) #This will find the corresponding k_s for each AR
            crit_stress = np.pi**2 * k_s * E /(12*(1-v**2)) * (t_sparweb/b)**2 #This will find the critical stresses for a given k_s
            k_s_z_back.append(k_s)   
            crit_stress_z_back.append(crit_stress) # Stores the relevant critical stress in a list
        
#         return crit_stress_z_back

    def front_spar_web_buckling_plot(self, wingspan):
        wingspan = self.wingspan
        z_values = np.linspace(0, wingspan/2, 100)
        for i in range(len(z_values)):
            plt.plot(z_values, self.front_spar_web_buckling())

#     def back_spar_web_buckling_plot(self, wingspan):
#         wingpan = self.wingspan
#         z_values = np.linspace(0, wingspan/2, 100)
#         for i in range(len(z_values)):
#             plt.plot(z_values, self.rear_spar_web_buckling())


    def margin_of_safety(self, z , V, k_v, AR, E, t_sparweb, b, v):
        critical_front = self.front_spar_web_buckling
        critical_rear = self.rear_spar_web_buckling
        mos_front_list = []
        mos_rear_list = []
        for i in range(len(critical_front)):
            a, b, h, alpha = self.geometry(z) # a and b not related to a_over_b
            #t_f = ###add function### 
            #t_r = ###add function### 
            #T_front = ###add function### 
            #T_rear = ###add function###
            avg_shear_front = V / (a * t_f + b * t_r) # formula for average shear
            max_shear_front = k_v * avg_shear_front # formula for maximum shear
            avg_shear_rear = V / (a * t_f + b * t_r)
            max_shear_rear = k_v * avg_shear_rear
            A = (a+b)*h/2 #enclosed area of trapezoical wingbox
            q_torsion_front = T_front / 2 / A #torsion shear stress in thin-walled closed section
            q_torsion_rear = T_rear / 2 / A
            mos_front = critical_front[i] / (max_shear_front + q_torsion_front * t_f)
            mos_rear= critical_rear[i] / (max_shear_rear + q_torsion_rear * t_r)
            mos_front_list.append(mos_front)
            mos_rear_list.append(mos_rear)
        return mos_front_list , mos_rear_list
    
#     Kc = interp_function(aspect_ratio)
#     print(f"Interpolated value at AR={x_query}: kc={y_query}")
#     return Kc

#         #everything under this part relates to stringer buckling:
# """"
# Note to self: 3 designs, so: 3 Areas and 3 I's 
# """"
# Area5 = 30e-3*3e-3 #Only one block, not entire area of L-stringer. area should be 90e-6: I dimensions translated into base and height of 30e-3 and thickness of 3e-3 
# Area8 = 40e-3*3.5e-3 #Only one block, not entire area of L-stringer. area should be 140e-6: I dimensions translated into base and height of 35e-3 and thickness of 4e-3
# Area9 = 30e-3*3e-3 #Only one block, not entire area of L-stringer. this is fine, option 9 was L stringer to begin with
# K = 1/4 #1 end fixed, 1 end free 

# #calculation of length: 
# #8 stringers on one side (take configuration with most stringers)
# #conservative estimate: take the longest stringer also !conservative estimate assumption: from root. Highest Length results in lowest critical stress
# #angle_stringer= 26.59493069 degrees at 1/9 of chord
# L = 15.04148123 #so 13.45 divided by cos(26.5949)
# #doublecheck value

# #centroid coordinates:
# x5_9=7.5e-3 #coordinates for option 5 and 9
# y5_9=7.5e-3#coordinates for option 5 and 9

# x_8= 10e-3
# y_8= 10e-3

# def Stringer_MOM ():#MoM around own centroid of L-stringer (bending around x-axis). So translate areas of I-stringer into L stringer. Also thin-walled assumption
#     I5 = 2*(Area5*x5_9**2)

#     I8 = 2*(Area8*x_8**2)

#     I9 = 2*(Area9*x5_9**2)

#     return I5, I8, I9

# def Stringer_buckling (E, K, L, I5,I8,I9): #critical stress of 3 different designs
#     stresscr_stringer_5= (K*np.pi**2*E*I5)/(L**2*(2*Area5))

#     stresscr_stringer_8= (K*np.pi**2*E*I8)/(L**2*(2*Area8))
    
#     stresscr_stringer_9= (K*np.pi**2*E*I9)/(L**2*(2*Area9))

#     return stresscr_stringer_5, stresscr_stringer_8, stresscr_stringer_9

#         #everything above this part relates to stringer buckling
 
   
    