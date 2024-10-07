import numpy as np
from scipy.optimize import minimize
from ISA import Density
from planform import V_inf
from ambiance import Atmosphere

atmosphere = Atmosphere(10668)
mtow = 37968  # Maximum takeoff weight in N
mlw = 0.886 * mtow  # Maximum landing weight in N
rho = atmosphere.density  # Air density
q = 0.5 * rho * V_inf**2  # Dynamic pressure

# Function to compute surface area from lift
def compute_surface_area(L, rho, V_inf, CL):
    return (2 * L) / (rho * V_inf**2 * CL)

def angle_at_xdivc(x, c, LEsweep, c_r, tr, b):
    return np.arctan(np.tan(LEsweep) - (x/c)  * 2 * (c_r/b) * (1-tr))

def gradient(AR, LE_sweep, tr):
    kin_visc = atmosphere.kinematic_viscosity
    L = 0.5 * (mtow + mlw) * 9.81  # Lift force (weight in N)
    
    # Calculate S dynamically based on required lift
    S = compute_surface_area(L, rho, V_inf, 0.56)
    
    # Calculate b from the relation
    b = np.sqrt(S * AR)

    # Define constants
    c = b / AR
    c_r = 2 * S / ((1 + tr) * b)

    # Calculate CL using the computed surface area S
    CL = 1.1 * (1 / q) * 0.5 * (mtow + mlw) * 9.81 / S
    
    # Efficiency factor (e) calculation
    e = 4.61 * (1 - 0.045 * AR**0.68) * (np.cos(LE_sweep)**0.15) - 3.1

    # Further calculations for drag coefficient
    cldes = 1.1 * (1/q) * 0.5 * (mtow + mlw) * 9.81 / S
    cdind = (cldes**2) / (np.pi * AR * e)
    
    S_wet_wing = 2 * 1.07 * S
    IF_wing = 1.4
    x_c_m = 0.37
    ttoc = 0.14
    lambda_m = angle_at_xdivc(37, 10, LE_sweep, c_r, tr, b)
    laminar_wing = 0.1
    k = 0.634e-5  # paint factor
    Re = V_inf * c / kin_visc
    # Laminar friction coefficient
    C_f_lamwing = 1.328 / np.sqrt(Re)
    # Turbulent friction coefficient
    C_f_turbwing = 0.455 / ((np.log10(Re)**(2.58)) * (1 + 0.144 * 0.77**2)**0.65)
    # Total friction coefficient weighted
    C_fwing = laminar_wing * C_f_lamwing + (1 - laminar_wing) * C_f_turbwing
    FF_wing = (1 + (0.6 / x_c_m) * ttoc + 100 * ttoc**4) * (1.34 * 0.77**0.18 * np.cos(lambda_m)**0.28)
    
    cd0 = (1/S) * (C_fwing * FF_wing * IF_wing * S_wet_wing)

    # Final drag coefficient
    cd = cd0 + cdind
    
    return cd, S, CL, e  # Return cd, S, CL, e


def objective(params):
    AR, LE_sweep, tr = params  # Remove b from the parameters
    cd, S, CL, e = gradient(AR, LE_sweep, tr)  # Get drag coefficient and surface area

    print(f"Objective Function - Params: {params}, Drag Coefficient: {cd}, Surface Area: {S}, Product: {S * cd}")
    taper_ratio_penalty = 0.1 * (0.2 - tr)
    # Minimize the product of S and cd
    return S * cd - 0.1 * AR  - taper_ratio_penalty# Introduce AR into the objective function with a negative weight to encourage maximization

# Constraint to ensure surface area is sufficient for lift
def lift_constraint(params):
    AR, LE_sweep, tr = params  # Remove b from the parameters
    L = mtow * 9.81  # Required lift force (aircraft weight in N)   
    
    # Calculate S based on lift and current parameters
    S = compute_surface_area(L, rho, V_inf, 0.56)  # Use a nominal CL for constraint
    _, S_current, CL, _ = gradient(AR, LE_sweep, tr)  # Get the current surface area
    return S_current - S  # Ensure the current S is sufficient

# Initial guesses for the parameters
initial_guess = [8.69, np.radians(27.9), 0.312]  # [AR, LE_sweep, taper ratio]

# Define bounds for the parameters
bounds = [
    (8, 12),  # AR bounds
    (np.radians(27.9), np.radians(35)),  # LE_sweep bounds (in radians)
    (0.2, 0.5)  # Taper ratio bounds
]

# Define the constraints
constraints = [{'type': 'ineq', 'fun': lift_constraint}]  # Inequality constraint

# Use SciPy's minimize function with bounds and constraints
result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints, method='SLSQP')

# Optimized parameters
optimized_params = result.x
optimized_cd, optimized_S, optimized_CL, optimized_e = gradient(*optimized_params)  # Calculate all values using optimized parameters
D = 0.5 * Density(10668) * V_inf**2 * (optimized_S) * optimized_cd
tsfc = 15.6e-6

SAR = V_inf/(D*tsfc)



# Print results
print("Optimized parameters (AR, LE_sweep, taper ratio):", optimized_params[0], np.degrees(optimized_params[1]), optimized_params[2])
print("Minimum Cd achieved:", optimized_cd)
print("Minimum product of S and drag coefficient:", result.fun)
print("Optimized wing surface area:", optimized_S[0])
print("Final lift coefficient (CL):", optimized_CL[0])
print("Final efficiency factor (e):", optimized_e)
print("The optimized SAR value is: ", SAR[0])