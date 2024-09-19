import pandas as pd
import math


filename = "airfoils/csv/NACA2212.csv"
df = pd.read_csv(filename)
cl_max = df['CL'].max()

ttochord = int(filename[19:21])/100

deltay = 26*ttochord
m = .8
delta = -0.32

CL_max = m*cl_max+delta

print(f"deltay is: {deltay}")
print(f"cl_max of airfoil is: {cl_max}")
print(f"CL_max of wing is: {CL_max}")