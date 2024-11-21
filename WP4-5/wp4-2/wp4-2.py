import scipy as sc
import numpy as np
from variables import *

class WingBox():
    def __init__(self, c_t, c_r, t):
        self.c_t = c_t
        self.c_r = c_r
        self.t = t
        
    
    def chord(self, z):
        c = self.c_r - self.c_r*(1-(self.c_t/self.c_r))*z
        return(c)

    def geometry(self, z):
        a = 0.1013 * self.chord(z)
        b = 0.0728 * self.chord(z)
        h = 0.55 * self.chord(z)
        return a, b, h
        
    def Torsion(self, z, T): # T : torsion,
        a, b, h = self.geometry(z)
        
        A = h * ( a + b ) / 2
        alpha = np.arctan(((a-b)/2)/h)
        S = a + b + 2 * (h / np.cos(alpha))
        thetadot = (T * S) / (4 * A * self.t * G)

        theta = sc.integrate.quad(thetadot, 0, z)
    
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

wingbox.chor