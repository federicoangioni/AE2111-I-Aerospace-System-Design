import sympy as sp

# Define the symbols
x = sp.symbols('x')

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


