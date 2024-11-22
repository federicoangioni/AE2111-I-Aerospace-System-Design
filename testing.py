import os

folder = 'WP4-5'
print(os.path.join(folder, os.listdir(folder)[0]))
print([os.path.join(folder, file) for file in os.listdir(folder)])