import numpy as np
from scipy.optimize import minimize
from ISA import Density
from planform import V_inf
from ambiance import Atmosphere
import sympy as sp

# Atmosphere initialization
atmosphere = Atmosphere(10668)
mtow = 37968  # Maximum takeoff weight in N
mlw = 0.886 * mtow  # Maximum landing weight in N
rho = atmosphere.density  # Air density
q = 0.5 * rho * V_inf**2  # Dynamic pressure

# Fixed parameters
LE_sweep = np.radians(27.9)  # Leading-edge sweep angle in radians
tr = 0.312  # Taper ratio

def angle_at_xdivc(x, c, LEsweep, c_r, tr, b):
    return np.arctan(np.tan(LEsweep) - (x/c) * 2 * (c_r/b) * (1-tr))

# Function to compute surface area from lift
def compute_surface_area(L, rho, V_inf, CL):
    return (2 * L) / (rho * V_inf**2 * CL)

def gradient(AR):
    L = 0.5 * (mtow + mlw) * 9.81  # Lift force (weight in N)
    rho = atmosphere.density  # Air density
    V_inf = 228.31123834310043  # Free-stream velocity in m/s

    # Calculate S dynamically based on required lift
    S = compute_surface_area(L, rho, V_inf, 0.56)
    c = np.sqrt(S * AR)  # Calculate chord length based on AR
    c_r = 2 * S / ((1 + tr) * np.sqrt(S * AR))

    CL = 1.1 * (1 / q) * 0.5 * (mtow + mlw) * 9.81 / S

    # Further calculations for drag coefficient
    cldes = 1.1 * (1/q) * 0.5 * (mtow + mlw) * 9.81 / S
    e = 4.61 * (1 - 0.045 * AR**0.68) * (np.cos(LE_sweep)**0.15) - 3.1
    cdind = (cldes**2) / (np.pi * AR * e)

    S_wet_wing = 2 * 1.07 * S
    IF_wing = 1.4
    x_c_m = 0.37
    ttoc = 0.14
    lambda_m = angle_at_xdivc(37, 10, LE_sweep, c_r, tr, np.sqrt(S * AR))
    laminar_wing = 0.1
    k = 0.634e-5  # Paint factor
    Re = V_inf * c / atmosphere.kinematic_viscosity

    # Laminar friction coefficient
    C_f_lamwing = 1.328 / np.sqrt(Re)
    # Turbulent friction coefficient
    C_f_turbwing = 0.455 / ((np.log10(Re)**2.58) * (1 + 0.144 * 0.77**2)**0.65)
    # Total friction coefficient weighted
    C_fwing = laminar_wing * C_f_lamwing + (1 - laminar_wing) * C_f_turbwing
    FF_wing = (1 + (0.6 / x_c_m) * ttoc + 100 * ttoc**4) * (1.34 * 0.77**0.18 * np.cos(lambda_m)**0.28)
    cd0 = (1 / S) * (C_fwing * FF_wing * IF_wing * S_wet_wing)

    # Final drag coefficient
    cd = cd0 + cdind

    return cd, S, CL, e

def objective(AR):
    cd, S, CL, e = gradient(AR)  # Get drag coefficient, surface area, lift coefficient, and e

    # Print current values for debugging
    print(f"Objective Function - AR: {AR}, Drag Coefficient: {cd}, Surface Area: {S}, Product: {S * cd}")

    # Minimize the product of S and cd while encouraging a higher AR
    return S * cd - 0.1 * AR  # Negative weight for AR to encourage maximization

# Constraint to ensure surface area is sufficient for lift
def lift_constraint(AR):
    L = 37968 * 9.81  # Required lift force (aircraft weight in N)
    rho = atmosphere.density  # Air density
    V_inf = 228.31123834310043  # Free-stream velocity

    # Get the current surface area
    _, S, _, _ = gradient(AR)  # Unpack all values from gradient

    CL = 1.1 * (1 / q) * 0.5 * (mtow + mlw) * 9.81 / S
    S_required = compute_surface_area(L, rho, V_inf, CL)  # Required surface area
    return S - S_required  # S must be greater than or equal to S_required

def wing_fuel_volume(AR):
    # Define the symbols
    x = sp.symbols('x')

    # Define constants
    AR = 10.41
    d = 0.31

    # Define the left-hand side of the equation
    lhs = (sp.sqrt(x * AR) / 3) * (
        0.0479 * (
            (2 * x / (sp.sqrt(x * AR) * (1 + d)))**2 + 
            (2 * x * d / (sp.sqrt(x * AR) * (1 + d)))**2
        ) + sp.sqrt(
            0.0479**2 * (2 * x / (sp.sqrt(x * AR) * (1 + d)))**4 * d**2
        )
    )

    # Define the right-hand side of the equation
    rhs = 10.223

    # Set up the equation
    equation = sp.Eq(lhs, rhs)

    # Solve for x
    S_required = sp.solve(equation, x)

    # Get the current surface area from the gradient function
    _, S, _, _ = gradient(AR)
    print(S_required
          )
    return S - S_required[0]  # Return the difference, ensuring S is greater than or equal to S_required

# Initial guess for AR
initial_guess = [10]  # [AR]

# Define bounds for AR
bounds = [(1, 14)]  # AR bounds

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

# Compute SAR
D = 0.5 * Density(10668) * V_inf**2 * (optimized_S) * optimized_cd
tsfc = 15.6e-6
SAR = V_inf / (D * tsfc)

Cldes_M077 = (optimized_CL / (np.cos(LE_sweep)**2))
Cldes_M0 = Cldes_M077 * np.sqrt(1 - 0.77**2)

# Print results
print("Optimized Aspect Ratio (AR):", optimized_AR)
print("Minimum drag coefficient:", optimized_cd)
print("Optimized wing surface area:", optimized_S)
print("Optimized lift coefficient (CL):", optimized_CL)
print("Airfoil lift coefficient then must be (Cldes)", Cldes_M0)
print("Optimized efficiency factor (e):", optimized_e)
print("The optimized SAR value is:", SAR[0])
