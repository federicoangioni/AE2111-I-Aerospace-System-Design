import numpy as np
from scipy.optimize import minimize
from ambiance import Atmosphere
import sympy as sp
from planform import LESweep, V_inf, taper_ratio
from variablesold import MTOW
import pandas as pd

# Atmosphere initialization
atmosphere = Atmosphere(10668)

mtow = 34903.81669975931
mlw = 0.886 * mtow  # Maximum landing weight in kg
rho = atmosphere.density  # Air density
q = 0.5 * rho * V_inf**2  # Dynamic pressure

df = pd.DataFrame([])
# Fixed parameters
tr = taper_ratio  # Taper ratio
# Function to calculate sweep at a given x/c location
def angle_at_xdivc(x, c, LESweep, c_r, tr, b):
    return np.arctan(np.tan(LESweep) - (x/c) * 2 * (c_r/b) * (1-tr))

# Function to compute surface area from lift
def compute_surface_area(L, rho, V_inf, CL):
    return (2 * L) / (rho * V_inf**2 * CL)

# Gradient function with iterative update for CL and S
def gradient(AR, tol=1e-6, max_iter=100, V_inf = V_inf):
    # Constants
    L = 0.5 * (mtow + mlw) * 9.81  # Lift force (weight in N)
    rho = atmosphere.density  # Air density
    # Free-stream velocity in m/s
    
    # Initial guess for surface area and lift coefficient
    CL = 0.5  # Initial guess for lift coefficient
    S = compute_surface_area(L, rho, V_inf, CL)  # Initial guess for surface area based on CL

    # Iteratively update S and CL until convergence
    for _ in range(max_iter):
        # Update CL based on the current surface area
        new_CL = L / (0.5 * rho * V_inf**2 * S)  # Dynamic lift coefficient based on current S
        
        # Recalculate surface area based on the new CL
        new_S = compute_surface_area(L, rho, V_inf, new_CL)

        # Check for convergence (difference between iterations is small)
        if abs(new_CL - CL) < tol and abs(new_S - S) < tol:
            break
        
        # Update CL and S for the next iteration
        CL = new_CL
        S = new_S

    # After convergence, proceed with the rest of the drag and efficiency calculations
    c = np.sqrt(S * AR)  # Calculate chord length based on AR
    c_r = 2 * S / ((1 + tr) * np.sqrt(S * AR))

    # Induced drag and efficiency factor calculations
    cldes = CL  # Use dynamically computed CL
    e = 4.61 * (1 - 0.045 * AR**0.68) * (np.cos(LESweep)**0.15) - 3.1
    cdind = (cldes**2) / (np.pi * AR * e)

    # Calculate wet area and friction drag
    S_wet_wing = 2 * 1.07 * S
    IF_wing = 1.4
    x_c_m = 0.37
    ttoc = 0.14
    lambda_m = angle_at_xdivc(37, 10, LESweep, c_r, tr, np.sqrt(S * AR))
    laminar_wing = 0.1
    Re = V_inf * c / atmosphere.kinematic_viscosity

    # Laminar and turbulent friction coefficients
    C_f_lamwing = 1.328 / np.sqrt(Re)
    C_f_turbwing = 0.455 / ((np.log10(Re)**2.58) * (1 + 0.144 * 0.77**2)**0.65)
    C_fwing = laminar_wing * C_f_lamwing + (1 - laminar_wing) * C_f_turbwing

    # Form factor and drag coefficient
    FF_wing = (1 + (0.6 / x_c_m) * ttoc + 100 * ttoc**4) * (1.34 * 0.77**0.18 * np.cos(lambda_m)**0.28)
    cd0 = (1 / S) * (C_fwing * FF_wing * IF_wing * S_wet_wing)

    # Final drag coefficient
    cd = cd0 + cdind
    #print(cd0)
    return cd, S, CL, e

# Objective function to minimize S * cd - 2 * AR
def objective(AR):
    cd, S, CL, e = gradient(AR)  # Get drag coefficient, surface area, lift coefficient, and e

    # Print current values for debugging
    #print(f"Objective Function - AR: {AR}, Drag Coefficient: {cd}, Surface Area: {S}, Product: {S * cd}")
    # Minimize the product of S and cd while encouraging a higher AR
    return S * cd - 2 * AR  # Negative weight for AR to encourage maximization

# Constraint to ensure surface area is sufficient for lift
def lift_constraint(AR, V_inf = V_inf ):
    L = mtow * 9.81  # Required lift force (aircraft weight in N)
    rho = atmosphere.density  # Air density

    # Get the current surface area
    _, S, _, _ = gradient(AR)  # Unpack all values from gradient

    CL = 1.1 * (1 / q) * 0.5 * (mtow + mlw) * 9.81 / S
    S_required = compute_surface_area(L, rho, V_inf, CL)  # Required surface area
    #print(f"REquired surfracea area for lift {S_required}")
    return S - S_required  # S must be greater than or equal to S_required


# Fuel volume constraint function
def wing_fuel_volume(AR):
    # AR comes in as a NumPy array, extract the scalar value
    AR = float(AR[0])  # Convert AR to a scalar value

    # Constants and symbols for SymPy
    x = sp.symbols('x')
    d = 0.31  # Taper ratio for fuel calculation (fixed)

    # Define the left-hand side of the equation (fuel volume equation)
    lhs = (sp.sqrt(x * AR) / 3) * (
        0.0479 * (
            (2 * x / (sp.sqrt(x * AR) * (1 + d)))**2 + 
            (2 * x * d / (sp.sqrt(x * AR) * (1 + d)))**2
        ) + sp.sqrt(
            0.0479**2 * (2 * x / (sp.sqrt(x * AR) * (1 + d)))**4 * d**2
        )
    )

    # Define the right-hand side of the equation (fuel volume requirement)
    rhs = 10.223

    # Set up the equation to solve for required surface area `S_required`
    equation = sp.Eq(lhs, rhs)

    # Solve the symbolic equation for `x`
    S_required = sp.solve(equation, x)

    # Ensure S_required is a valid real solution
    if isinstance(S_required, list) and len(S_required) > 0:
        # Filter out any complex or invalid solutions, and take the first valid real solution
        S_required = [sol.evalf() for sol in S_required if sol.is_real and sol > 0]
        if len(S_required) == 0:
            raise ValueError("No valid solution found for required surface area.")
        S_required = float(S_required[0])  # Take the first valid solution as a float
    else:
        raise ValueError("Unable to solve for required surface area.")

    # Get the actual surface area from the gradient function
    _, S, _, _ = gradient(AR)
    #print(f"The required wing area for fuel volume: {S_required}")
    # Return the difference (S must be greater than or equal to S_required)
    return S - S_required

# Initial guess for AR
initial_guess = [10]  # [AR]

# Define bounds for AR
bounds = [(5, 14)]  # AR bounds

# Define the constraints
constraints = [
    {'type': 'ineq', 'fun': lift_constraint},
    {'type': 'ineq', 'fun': wing_fuel_volume}  # New constraint from the equation
]

# Use SciPy's minimize function with bounds and constraints
result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints, method='SLSQP')

# Optimized parameters
optimized_AR = result.x[0]
optimized_cd, optimized_S, optimized_CL, optimized_e = gradient(optimized_AR)  # Calculate drag coefficient, surface area, CL, and e using optimized parameters
optimized_S  = optimized_S[0]
# Compute SAR
D = 0.5 * rho * V_inf**2 * (optimized_S) * optimized_cd
tsfc = 15.6e-6
SAR = V_inf / (D * tsfc)

Cldes_M077 = (optimized_CL / (np.cos(LESweep)**2))
Cldes_M0 = Cldes_M077 * np.sqrt(1 - 0.77**2)

# New values computing
optimized_b = np.sqrt(optimized_AR*optimized_S) # m

if __name__ == "__main__":
    print("Optimized Aspect Ratio (AR):", optimized_AR)
    print("Minimum drag coefficient:", optimized_cd)
    print("Optimized wing surface area:", optimized_S)
    print("Optimized lift coefficient (CL):", optimized_CL)
    print("Airfoil lift coefficient then must be (Cldes)", Cldes_M0)
    print("Optimized efficiency factor (e):", optimized_e)
    print("The optimized SAR value is:", SAR)