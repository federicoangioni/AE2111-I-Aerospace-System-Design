import numpy as np
from ambiance import Atmosphere

# Atmosphere initialization
altitude = 10668  # Altitude in meters
atmosphere = Atmosphere(altitude)
mtow = 37968 * 9.81  # Maximum takeoff weight in N (mass in kg * gravity)
mlw = 0.886 * mtow  # Maximum landing weight in N
rho = atmosphere.density[0]  # Air density (kg/m^3)
V_inf = 228.31123834310043  # Free-stream velocity in m/s
q = 0.5 * rho * V_inf**2  # Dynamic pressure

# Fixed parameters
LE_sweep = np.radians(27.9)  # Leading-edge sweep angle in radians
tr = 0.312  # Taper ratio

# Function to calculate surface area given lift, air density, velocity, and CL
def compute_surface_area(L, rho, V_inf, CL):
    return (2 * L) / (rho * V_inf**2 * CL)

# Function to calculate the geometric angle at a point along the wing
def angle_at_xdivc(x, c, LEsweep, c_r, tr, b):
    return np.arctan(np.tan(LEsweep) - (x / c) * 2 * (c_r / b) * (1 - tr))

# Gradient function that dynamically updates surface area and CL
def gradient(AR, tolerance=1e-6, max_iterations=100):
    # Average lift force based on mtow and mlw
    L = 0.5 * (mtow + mlw)  
    rho = atmosphere.density[0]
    V_inf = 228.31123834310043
    q = 0.5 * rho * V_inf**2

    # Set an initial guess for CL
    CL = 0.6  # Initial guess for lift coefficient
    S = compute_surface_area(L, rho, V_inf, CL)  # Initial guess for S based on initial CL

    iteration = 0
    delta = float('inf')  # Initialize the change between iterations

    # Iteratively update S and CL until they converge
    while delta > tolerance and iteration < max_iterations:
        # Calculate new CL based on current surface area S
        CL_new = L / (q * S)
        # Calculate new S based on the updated CL
        S_new = compute_surface_area(L, rho, V_inf, CL_new)

        # Check how much S has changed between iterations
        delta = abs(S_new - S)

        # Update S and CL for the next iteration
        S = S_new
        CL = CL_new
        iteration += 1

    # Proceed with the rest of the calculations using the final converged S and CL
    c = np.sqrt(S / AR)  # Mean aerodynamic chord
    b = np.sqrt(AR * S)  # Wingspan
    c_r = (2 * S) / ((1 + tr) * b)  # Root chord length

    # Calculate mid-chord sweep angle lambda_m
    x_c_m = 0.37 * c  # x/c location for mean aerodynamic chord
    lambda_m = angle_at_xdivc(x_c_m, c, LE_sweep, c_r, tr, b)

    # Desired lift coefficient
    cldes = CL
    e = 4.61 * (1 - 0.045 * AR**0.68) * (np.cos(LE_sweep)**0.15) - 3.1
    cdind = (cldes**2) / (np.pi * AR * e)

    # Wet wing surface area and interference factor
    S_wet_wing = 2 * 1.07 * S
    IF_wing = 1.4  # Interference factor
    ttoc = 0.14  # Thickness-to-chord ratio
    M = 0.77  # Mach number

    # Estimate Reynolds number
    laminar_wing = 0.1  # Fraction of laminar flow
    Re = V_inf * c / atmosphere.kinematic_viscosity[0]

    # Laminar and turbulent friction coefficients
    C_f_lamwing = 1.328 / np.sqrt(Re)
    C_f_turbwing = 0.455 / ((np.log10(Re)**2.58) * (1 + 0.144 * M**2)**0.65)
    C_fwing = laminar_wing * C_f_lamwing + (1 - laminar_wing) * C_f_turbwing

    # Form factor and drag coefficient
    FF_wing = (1 + (0.6 / 0.37) * ttoc + 100 * ttoc**4) * (1.34 * M**0.18 * np.cos(lambda_m)**0.28)
    cd0 = (1 / S) * (C_fwing * FF_wing * IF_wing * S_wet_wing)

    # Final drag coefficient
    cd = cd0 + cdind

    return cd, S, CL, e

# Function to enforce the lift constraint
def enforce_lift_constraint(AR):
    L = mtow  # Lift force must equal maximum takeoff weight
    rho = atmosphere.density[0]
    V_inf = 228.31123834310043
    q = 0.5 * rho * V_inf**2

    # Compute S and CL dynamically for the given AR
    _, S, CL, _ = gradient(AR)

    # Required CL for lift
    CL_required = L / (q * S)

    # Check if the computed CL meets or exceeds the required CL for lift
    if CL < CL_required:
        # Adjust S dynamically to ensure the lift constraint is met
        S_new = compute_surface_area(L, rho, V_inf, CL_required)
        return S_new  # Return the updated surface area after enforcing the constraint
    return S

# Function to calculate the gradient of the drag coefficient with respect to AR
def calculate_gradient_cd(AR, step=0.01):
    # Compute the drag coefficient for AR and AR + step
    cd, S, CL, e = gradient(AR)
    cd_step, S_step, CL_step, e_step = gradient(AR + step)

    # Approximate gradient (change in cd divided by step)
    gradient_cd = (cd_step - cd) / step
    return gradient_cd, cd

# Gradient descent algorithm to minimize cd with respect to AR
def gradient_descent_cd(initial_AR, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
    AR = initial_AR  # Start with the initial guess
    iteration = 0
    while iteration < max_iterations:
        grad_cd, current_cd = calculate_gradient_cd(AR)

        # Update AR using the gradient of the drag coefficient
        AR_new = AR - learning_rate * grad_cd

        # Enforce the lift constraint on the new AR
        S_new = enforce_lift_constraint(AR_new)
        
        # Check for convergence
        if abs(AR_new - AR) < tolerance:
            break
        
        AR = AR_new
        iteration += 1

        # Print debug information for each iteration
        print(f"Iteration {iteration}, AR: {AR}, Drag Coefficient: {current_cd}, Gradient: {grad_cd}")
    
    return AR

# Set initial guess for AR
initial_AR = 10  # Initial guess for AR

# Run gradient descent to minimize cd and optimize AR
optimized_AR = gradient_descent_cd(initial_AR)

# After optimization, calculate the optimized aerodynamic properties
optimized_cd, optimized_S, optimized_CL, optimized_e = gradient(optimized_AR)

# Compute Specific Air Range (SAR)
D = 0.5 * rho * V_inf**2 * optimized_S * optimized_cd
tsfc = 15.6e-6  # Thrust specific fuel consumption in kg/(N·s)
SAR = V_inf / (D * tsfc)

Cldes_M077 = optimized_CL / (np.cos(LE_sweep)**2)
Cldes_M0 = Cldes_M077 * np.sqrt(1 - 0.77**2)

# Print results
print("Optimized Aspect Ratio (AR):", optimized_AR)
print("Minimum drag coefficient:", optimized_cd)
print("Optimized wing surface area:", optimized_S)
print("Optimized lift coefficient (CL):", optimized_CL)
print("Airfoil lift coefficient then must be (Cldes):", Cldes_M0)
print("Optimized efficiency factor (e):", optimized_e)
print("The optimized SAR value is:", SAR)
