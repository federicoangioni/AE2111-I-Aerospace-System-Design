import numpy as np

file_name = "MainWing_a=0.00_v=10.00ms.txt"

data = np.genfromtxt(
    file_name,
    skip_header=30,  # Adjust this based on where the spanwise coefficients start
    invalid_raise=False,
    delimiter=None,  # Space or tab delimiter
)

print(data)