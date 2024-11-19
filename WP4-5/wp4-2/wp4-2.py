import scipy as sc

class WingBox():
    def __init__(self):
        pass
    
    def chord(self, z, c_r, c_t):
        c = c_r - c_r*(1-(c_t/c_r))*z
        return(c)
    
    def Centroid(self, c, t):
        A1 = 0.1013*c*t
        A2 = 0.55*c*alpha*t
        A3 = 0.1013*c*t
        A4 = 0.1013*c*t

    def stiffener(self, number, type):
        type = ["L", "I", ]
        
        
        
        
wingbox = WingBox()
