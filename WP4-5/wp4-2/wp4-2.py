import scipy as sc
import numpy as np

class WingBox(self):
    
    def chord(z, c_r, c_t):
        c = c_r - c_r*(1-(c_t/c_r))*z
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
    def Centroid(self, c, t, alpha, stringer_x_pos[], stringer_y_pos[], stringer_area[]):
        A = [0.0728*c*t, 0.1013*c*t, 0.55*c*np.sin(np.radians(alpha))*t, 0.55*c*np.sin(np.radians(alpha))*t] #Areas of the components
        X = [0, 0.55*c, 0.5*0.55*c*np.cos(np.radians(alpha)), 0.5*0.55*c*np.cos(np.radians(alpha))] # X positions of the components
        Y = [0, 0, 0.5*0.1013*c-0.5*0.55*c*np.sin(np.radians(alpha)), -0.5*0.1013*c+0.5*0.55*c*np.sin(np.radians(alpha))] # Y positions of the components
        while j <= len(stringer_x_pos): #include the contributions of the stringers
            A.append(stringer_area[j])
            X.append(stringer_x_pos[j])

        while i <= len(X):
            weights = A[i]*A[X]
        

        x = (weights)/sum(A)

    def MOMEWB (self,):

        x = (weights)/sum(A)
    
    def I_stiffeners(self, type: str, dimensions: list):
        type = ["L", "I"]
        
        """
        dimensions: changes in base of the used stringer
        L type stringer: [base, height, thickness base, thickness height]
        """
        
        
        
        




wingbox = WingBox()

