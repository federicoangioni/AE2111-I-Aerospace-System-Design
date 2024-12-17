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
        
        
        


class RibWebBuckling():
    print("hello world")



























    def back_spar_web_buckling(self, a_over_b, E, t_sparweb, b):
        k_s_array_np = np.array(k_s_array)
        ab_values = k_s_array_np[:, 0]
        k_s_values = k_s_array_np[:, 1]
        k_s = np.interp(a_over_b, ab_values, k_s_values)
        crit_stress = np.pi**2 * k_s * E /(12*(1-0.33**2)) * (t_sparweb/b)**2
        print('Critical stress is', crit_stress)