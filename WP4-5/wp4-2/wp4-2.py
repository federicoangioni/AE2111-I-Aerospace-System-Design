import scipy as sc

class WingBox(self):
    
    def chord(z, c_r, c_t):
        c = c_r - c_r*(1-(c_t/c_r))*z
        return(c)
    
    def Centroid(c, t):
        A1 = 0.1013*c*t
        A2 = 0.55*c*alpha*t
        A3 = 0.1013*c*t
        A4 = 0.1013*c*t

wingbox = WingBox()
