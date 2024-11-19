import scipy as sc
def chord(z, c_r, c_t):
    c = c_r - c_r*(1-(c_t/c_r))*z
    return(c)
