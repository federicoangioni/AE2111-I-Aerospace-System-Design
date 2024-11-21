import scipy as sc
import numpy as np

class WingBox():
    def __init__(self, c_t, c_r):
        self.c_t = c_t
        self.c_r = c_r
        
    
    def chord(self, z):
        c = self.c_r - self.c_r*(1-(self.c_t/self.c_r))*z
        return(c)

    def geometry(self, z):
        a = 0.1013 * self.chord(z)
        b = 0.0728 * self.chord(z)
        h = 0.55 * self.chord(z)
        return a, b, h
        
    def Polar(self, t, alpha, z): # t : thickness,
        a, b, h = self.geometry(z)
        
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
    
    def I_stiffeners(self, type: str, dimensions: dict):
        type = ["L", "I"]
        
        """
        dimensions: changes in base of the used stringer
        L type stringer: {base, height, thickness base, thickness height}
        I type stringer: [base, top, web height, thickness top, thickness web, thickness bottom]
        """
        if type == "L":
            x = (dimensions["base"]*(dimensions["thickness base"]**2)/2)/(dimensions["base"]*dimensions["height"]*dimensions["thickness base"]*dimensions["thickness height"])
            y = (dimensions["height"]**2 * dimensions["thickness height"]/2) / (dimensions["base"]*dimensions["height"] + dimensions["thickness base"]*dimensions["thickness height"])
            
            I_xx = 0 
            I_yy = 0
            A = dimensions["base"]*dimensions["height"] + dimensions["thickness base"]*dimensions["thickness height"]
        
        elif type == "I":
            pass
        
        




wingbox = WingBox()

wingbox.chor