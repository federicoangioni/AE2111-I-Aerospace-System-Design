import sympy as sp
from math import sqrt

# Declare symbolic variable for y
y = sp.symbols('y')

# Parameters
b = 24.86           # Span
S_ref = 69.7        # Reference wing area
c_d0 = 0.01       # Drag coefficient at zero lift (Airfoil)
cl_alpha = 0.1236    # Lift curve slope (Airfoil)
deltaa = 22.5        # Aileron deflection angle 
labda = 0.312     # Taper ratio
tau = 0.42           # Control effectiveness factor
b1 = 0.65 * (b / 2)   # Inner boundary of aileron
b2 = 0.9 * (b / 2)   # Outer boundary of aileron
Cr = 4.27            # Root chord length
W = 376301.8         # MTOW in kg
g = 9.81    #Gravitational constant in m/s^2
rho = 1.225  # Air density at sea level
C_l_max = 1.44 #C_l_max for airfoil
V_sr1 =  sqrt((W)/S_ref * 2/rho * 1/C_l_max)        # V_sr1 Approach speed at Sea level
V = 1.13 * V_sr1     # including safety factor according to CS25 (not included for ADSEE II)
bankang = 45         # Bank Angle


# Equation for local chord (linear taper assumption)
C_y = Cr * (1 + 2 * ((labda - 1) / b)*y)

# Equation for aileron control derivative
Cl_delta_a = (2 * cl_alpha * tau / (S_ref * b)) * sp.integrate(C_y * y, (y, b1, b2))

# Equation for roll damping coefficient
C_L_p = -(4 * (cl_alpha + c_d0) / (S_ref * b**2)) * sp.integrate(C_y * y^2, (y, 0, (b/2))) #be sure to replace y^2 with y double star, like with b
# Equation for steady-state roll rate
P = - (Cl_delta_a/C_L_p) * deltaa * ((2*V)/b)

# Equation for delta t
delta_t = bankang/P

#Area Aileron
A_aileron = 0.2 * sp.integrate(C_y, (y, b1, b2))



# Output results
print(f"Aileron control derivative (Cl_delta_a): {Cl_delta_a}")
print(f"Roll damping coefficient (C_L_p): {C_L_p}")
print(f"Steady state roll rate (P): {P} [deg/s]")
print(f"Delta t (delta_t): {delta_t} [s]")
print(f"Area Aileron (A_aileron): {A_aileron} [m^2]")