import scipy as sc
import numpy as np

class WingBox():
    def __init__(self):
        pass
    
    def chord(self, z, c_r, c_t, b): #
        c = c_r - c_r*(1-(c_t/c_r))*z/(b/2)
        return(c)
    
    def Centroid(self, c, t, alpha):
        A1 = 0.0728*c*t
        A2 = 0.1013*c*t
        A3 = 0.55*c*np.sin(np.radians(alpha))*t
        A4 = A3

        x = (A1*0 + A2*0.55*c + 0.5*0.55*c*np.cos(np.radians(alpha)) + 0.5*0.55*c*np.cos(np.radians(alpha)))/(A1+A2+A3+A4)

    
    def Polar(self, )# t : thickness, 
    
    
    
    def Centroid(self, c, t, alpha, stringer_x_pos[], stringer_area[]):
        A = [0.0728*c*t, 0.1013*c*t, 0.55*c*np.sin(np.radians(alpha))*t, 0.55*c*np.sin(np.radians(alpha))*t] #area 
        X = [0, 0.55*c, 0.5*0.55*c*np.cos(np.radians(alpha)), 0.5*0.55*c*np.cos(np.radians(alpha))]

        while j <= len(stringer_x_pos)
            A.append(stringer_area[j])
            X.append(stringer_x_pos[j])

        while i <= len(X):
            weights = A[i]*A[X]
        

        x = (weights)/sum(A)
        
        




wingbox = WingBox()