import numpy as np
import pandas as pd      
import matplotlib.pyplot as plt
      

class SkinBuckling():
    def __init__(self, n_ribs, wingbox_geometry, wingspan):
        """
        wingbox_geometry: remember this is a function of z, it is given by WingBox.geometry(z)
        wingspan: # modified half wingspan from the attachement of the wing with the fuseslage to the tip, 
                    you can use WingBox.wingspan to obtain it, it has been defined like this even if it's a half span insult fede for this :)
        """        
        
        # attributing to class variable
        self.geometry = wingbox_geometry 
        
        # attributing to class variable
        self.halfspan = wingspan / 2
        
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
        
    def skin_AR(self):
        """
        The wing is defined with a rib at the root and at the tip orginally 
        """
        # nr of panels on the half wing will always be one less than the number of ribs
        n_panels = self.n_ribs - 1
        
        # here ASSUMPTION, the wing ribs are equally spaceed along the half span
        # length of one panel
        l_panel = self.halfspan/n_panels
        
        l_ribs = []
        panel_area = []
        panel_AR = []

        for i in range(self.n_ribs):
            print(i)
            a, b, h, alpha = self.geometry(l_panel * i)
            l_ribs.append(h)

        for i in range(0, n_panels+ 1):
            
            panel_area.append(0.5*l_panel*(l_ribs[i] + l_ribs[i+1]))
            AR = panel_area[i]/(l_panel ** 2)

            if AR >=1:
                panel_AR.append(AR)
            else:
                panel_AR.append(panel_area[i]/l_ribs[i]**2)

        # im not sure i understand this below by fede
    
        # for i in range(number_of_panels-1):
        #     if z >= length_of_the_panel * i and z<length_of_the_panel * (i+1):
        #         AR_final = panel_AR[i]
        #     else: AR_final = -100000
        
        print(l_ribs)
        return panel_AR
    
    
    def PlotSkinAR(number_of_ribs, wing_span, geometry, SkinAspectRatio):
        AR_final_values = []
        z_values = np.linspace(0, wing_span / 2, 100)

        # Calculate AR_final for each z value
        for z in z_values:
            AR_final = SkinAspectRatio(number_of_ribs, wing_span, geometry, z)
            AR_final_values.append(AR_final)

    # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(z_values, AR_final_values, marker='o', linestyle='-', color='b', label='AR_final(z)')
        plt.title("Skin Aspect Ratio as a Function of z")
        plt.xlabel("z Position")
        plt.ylabel("Aspect Ratio (AR_final)")
        plt.axhline(y=1, color='r', linestyle='--', label='AR = 1 Threshold')  # Add a reference line
        plt.grid(True)
        plt.legend()
        plt.tight_layout()  # Improve layout
        plt.show() 





#     def StressSkinBuckling(panel_AR, E, Poisson, constant, t, z):
#         stress = np.pi**2 * constant*E/(12*(1-Poisson**2))*(t/b)**2
#         return stress




class RibWebBuckling():
    def __init__(self, wingbox_geometry: function, wingspan):
        # attributing to class variable
        self.geometry = wingbox_geometry 
        
        # attributing to class variable
        self.wingspan = wingspan
        pass  

    def chord(self, z, c_r, c_t, wingspan): 
        # returns the chord at any position z in meters, not a percentage of halfspan, on 28/11 it can go from 0 to b/2 - intersection*b/2
        c = c_r - c_r * (1 - (c_t / c_r)) * (z / ((wingspan / 2)))
        return c
    
    # Defines AspectRaio of the long Spar
    def LongSparWebAR(self, z):
        wing_span = self.wingspan
        a = self.geometry(z)
        LongSparWebAR_z = []
        z_values = np.linspace(0,wing_span/2, 100) 
        for z in z_values:
            t_0 = a(0)
            t_1 = a(z_values)
            S = z * (t_0 + t_1) / 2    
            AR = (z**2)/S
            LongSparWebAR_z.append(AR)

        return LongSparWebAR_z
    
    def ShortSparWebAR(self, z):
        wing_span = self.wingspan
        b = self.geometry(z)
        ShortSparWebAR_z = []
        z_values = np.linspace(0,wing_span/2, 100) 
        for z in z_values:
            t_0 = b(0)
            t_1 = b(z_values)
            S = z * (t_0 + t_1) / 2    
            AR = (z**2)/S
            ShortSparWebAR_z.append(AR)
        
        return ShortSparWebAR_z

    
    def front_spar_web_buckling(self, AR, E, t_sparweb, b, v):
        from k_s_curve import k_s_array
        AR = self.LongSparWebAR()
        k_s_array_np = np.array(k_s_array)
        ab_values = k_s_array_np[:, 0]
        k_s_values = k_s_array_np[:, 1]
        crit_stress_z_front = []
        for i in AR:
            k_s = np.interp(AR, ab_values, k_s_values) #This will find the corresponding k_s for each AR
            for j in k_s:
                crit_stress = np.pi**2 * k_s * E /(12*(1-v**2)) * (t_sparweb/b)**2 #This will find the critical stresses for a given k_s
                crit_stress_z_front.append(crit_stress) # Stores the relevant critical stress in a list
        
        return crit_stress_z_front
    
    def front_spar_web_buckling_plot(self):
        wingspan = self.wingspan
        z_values = np.linspace(0, wingspan/2, 100)
        for i in range(len(z_values)):
            plt.plot(z_values, self.front_spar_web_buckling())

