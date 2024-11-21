import scipy as sc
import numpy as np
import math as math

class WingBox():
    def __init__(self):
        self.a = 
    
    def chord(self, z, c_r, c_t):
        c = c_r - c_r*(1-(c_t/c_r))*z
        return(c)

    
    def Polar(self, t, alpha): # t : thickness,
        A = 0.55 * self.chord() * (0.0728 * self.chord() + 0.1013 * self.chord())/2
        S = 0.0728 * self.chord() + 0.1013 * self.chord() + 2 * 0.55*c*np.cos(np.radians(alpha))
        J = (4 * t * A**2)/S 
        return(J)
    
    def Centroid(self, c, t, alpha, stringer_x_pos, stringer_y_pos, stringer_area):# c-chord, t-thickness, alpha-
        A = [0.0728*c*t, 0.1013*c*t, 0.55*c*np.sin(np.radians(alpha))*t, 0.55*c*np.sin(np.radians(alpha))*t] #Areas of the components
        X = [0, 0.55*c, 0.5*0.55*c*np.cos(np.radians(alpha)), 0.5*0.55*c*np.cos(np.radians(alpha))] # X positions of the components
        Y = [0, 0, 0.5*0.1013*c-0.5*0.55*c*np.sin(np.radians(alpha)), -0.5*0.1013*c+0.5*0.55*c*np.sin(np.radians(alpha))] # Y positions of the components

        while j <= len(stringer_x_pos): #include the contributions of the stringers
            A.append(stringer_area[j])
            X.append(stringer_x_pos[j])
            Y.append(stringer_y_pos[j])
            j+=1

        while i <= len(X):
            weights_X = A[i]*X[i]
            weights_Y = A[i]*Y[i]
            i+=1
        
        x = (weights_X)/sum(A)
        y= (weights_Y)/sum(A)
        
        return(x, y)

    def MOMEWB (self,):

        x = (weights)/sum(A)
    
    def I_stiffeners(self, type: str, dimensions: list):
        type = ["L", "I"]
        
        """
        dimensions: changes in base of the used stringer
        L type stringer: [base, height, thickness base, thickness height]
        I type stringer: [base, top, web height, thickness top, thickness web, thickness bottom]
        """
        if type == "L":
            I_xx = (1/12)*dimensions[2]**3*dimensions[4]
            I_yy = 0
        
        elif type == "I":
            pass
        
        




wingbox = WingBox()

