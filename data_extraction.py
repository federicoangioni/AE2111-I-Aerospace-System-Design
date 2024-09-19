import csv

filename = "airfoils/textfile/NACA2312.txt" # this file is the main file
naca_digits = 4
titles = 10 
csv_file = "airfoils/csv/NACA" + filename[22 : 22 + naca_digits] + ".csv"

with open(filename, 'r') as file:
    lines = file.readlines()
    columns = lines[titles-1].strip().split()

    lst = []
    for line in lines[11:]:
        line = line.strip().split()
        lst.append(line)        

with open(csv_file, 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(columns)
    csv_writer.writerows(lst)