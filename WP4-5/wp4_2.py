from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#general: assumption is symmetric wing box
class WingBox():
    def __init__(self, t: int, c_r: int, c_t: int, tr:int = None):
        self.c_t = tr * c_r if c_t is None else c_t # tip chord [m]
        self.c_r = c_r                              # root chord [m]
        self.t = t                                  # wingbox thickness, constant thickness in the cross sectiona nd along z assumed [m]
        self.deflections = pd.DataFrame(columns = ['Load [Nm]', 'z location [m]', 'Displacement [m]',
                                                    'Rotation [rad]', 'Moment of Inertia I [m^4]', 'Polar moment of Inertia J [m^4 (??)]'])
        
    def chord(self, z): 
        # returns the chord at any position z
        c = self.c_r - self.c_r*(1-(self.c_t/self.c_r))*z
        return c

    def geometry(self, z: int):
        # returns the wing box geometry (side lenghts) at any position z
        a = 0.1013 * self.chord(z) # trapezoid longer  [m]
        b = 0.0728 * self.chord(z) # trapezoid shorter [m]
        h = 0.55 * self.chord(z)   # trapezoid height  [m]
        return a, b, h
    
    def torsion (self, z, T: int, G): # T : torsion, 
        a, b, h = self.geometry(z)
        
        A = h * (a + b) / 2
        alpha = np.arctan((a - b) / (2 * h))
        S = a + b + 2 * (h / np.cos(alpha))
        thetadot = lambda z: (T * S) / (4 * A * self.t * G)

        theta = integrate.quad(thetadot, 0, z)

        return theta
    
    def bending (self, z, M, E):
        I = self.MOMEWB()
        v_double_dot = lambda z: M/(-E*I)
        
        vdot = integrate.quad(v_double_dot, 0, z)
        v = integrate.quad(vdot, 0, z)
        
        return v
    
    def centroid(self, z, stringer_x_pos, stringer_y_pos, stringer_area):# c-chord, t-thickness, alpha-
        
        a, b, h = self.geometry(z)
        alpha = np.arctan(((a-b)/2)/h)
        
        A = [b*self.t, a*self.t, h*np.sin(alpha)*self.t, h*np.sin(alpha)*self.t] #Areas of the components [longer side, shorter side, oblique, oblique]
        X = [0, h, 0.5*h*np.cos(alpha), 0.5*h*np.cos(alpha)]                     # X positions of the components
        Y = [0, 0, 0.5*a*-0.5*h*np.sin(alpha), -0.5*a+0.5*h*np.sin(alpha)]       # Y positions of the components

        while j <= len(stringer_x_pos): #include the contributions of the stringers
            A.append(stringer_area[j])
            X.append(stringer_x_pos[j])
            Y.append(stringer_y_pos[j])
            j+=1

        while i <= len(X): #calculate the weights
            weights_X = A[i]*X[i]
            weights_Y = A[i]*Y[i]
            i+=1
        
        x = weights_X/sum(A) #x position of the centroid
        y= weights_Y/sum(A)  #y position of the centroid
        
        return x, y

    def MOMEWB (self,z ,t, h, x, y): #Moment of inertia for empty wing box, #ci and cj are related to distance from centroid/coordinate system
        a, b, h = self.geometry(z)
        ci1= h - x
        ci2= x
        ci3= np.cos(np.radians(alpha))*((h/np.cos(np.radians(alpha)))/2) - y
        cj3= (b/2)+ np.sin(np.radians(alpha))*((h/np.cos(np.radians(alpha)))/2)

        #Split into 3 section: 1 is the short vertical bar, 2 is the long vertical bar, and 3 are the bars at an angle
        #section 1:
        I1xx = (t*b**3)/12 + 0 #I know it's silly
        I1yy = 0 + (t*b)*ci1**2

        #section 2:
        I2xx = (t*a**3)/12 + 0
        I2yy = 0 + (t*a)*ci2**2

        #section 3 (so both bars): #bar at angle practically same as bar: 0.5501845713 chord
        I3xx= 2/12*t*(np.sin(np.radians(alpha))**2)*((h/np.cos(np.radians(alpha)))**3) +2*((h/np.cos(np.radians(alpha)))*t)*cj3**2
        I3yy= 2/12*t*(np.cos(np.radians(alpha))**2)*((h/np.cos(np.radians(alpha)))**3) +2*((h/np.cos(np.radians(alpha)))*t)*ci3**2

        #Total moments of inertia of wing box:
        I_wingbox_xx = I1xx+I2xx+I3xx
        I_wingbox_yy = I1yy+I2yy+I3yy

        return(I_wingbox_xx,I_wingbox_yy)
    
    def I_stiffener(self, type: str, dimensions: dict, distance: tuple):
        type = ["L", "I"]
        
        """
        dimensions: changes in base of the used stringer
        L type stringer: {base, height, thickness base, thickness height}
        I type stringer: [base, top, web height, thickness top, thickness web, thickness base]
        distance: tuple, distance from centroid (x, y)
        """
        
        if type == "L":
            x = (dimensions["base"]*(dimensions["thickness base"]**2)/2)/(dimensions["base"]*dimensions["height"]*dimensions["thickness base"]*dimensions["thickness height"])
            y = (dimensions["height"]**2 * dimensions["thickness height"]/2) / (dimensions["base"]*dimensions["height"] + dimensions["thickness base"]*dimensions["thickness height"])
            
            I_xx = 0 
            I_yy = 0
            A = dimensions["base"]*dimensions["height"] + dimensions["thickness base"]*dimensions["thickness height"]
        
        elif type == "I":
            A = dimensions["base"]*dimensions["thickness base"] + dimensions["web"]*dimensions["thickness web"] + dimensions["top"]* dimensions["thickness top"]
        
        # returning only Steiner's terms for now
        return (distance[0] ** 2 * A, distance[1] ** 2 * A)
    
    def show(self, load, modulus, halfspan, choice: str): 
        """
        load: int function representing the internal load of the wing [N]
        modulus: either E or G depending on the analysis [N/m2]
        halfspan: halfspan length [m]
        choice: 'bending' or 'torsion', string input
        """
        type = ['bending', 'torsion']
        
        z = np.linspace(0, halfspan) # range of z values to show the plot
        
        I_xx = 0 # defining the moment of inertia
        
        J = 0 
        
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
