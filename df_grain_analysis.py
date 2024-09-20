# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 07:01:45 2024

@author: ramos
"""

import pandas as pd

# Input csv to read
input_csv = 'C:/Users/ramos/Desktop/coco/validation/report/output_clean.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(input_csv)

print(df.head())


# Remove '.tif' from the 'filename' column
df['filename'] = df['filename'].str.replace('.tif', '')

print(df.head())


print(df.info())


# pixel2 to mm2:
pixel2_to_mm2 = 0.0087

# # pixel to mm:
# pixel_to_mm = 0.0933


# Multiply columns 1, 2, and 3 by 0.0087
df[['area', 'filled_area', 'convex_area']] *= pixel2_to_mm2

# # Multiply columns 4 and 5 by 0.0933
# df[['major_axis_length', 'minor_axis_length']] *= 0.0933


# Describe the 'filled_area' column in the DataFrame df
print(df['convex_area'].describe())



# Assuming df is your DataFrame
max_df = df.groupby('filename')['convex_area'].max().reset_index()

# Display the new DataFrame
print(max_df)


# Assuming df is your DataFrame
rows = max_df[(max_df['convex_area'] > 600) & max_df['filename'].str.contains('limpo')]
unique_filenames = rows['filename'].unique()
print(unique_filenames)
print(len(unique_filenames))

# # Describe the 'filled_area' column in the DataFrame df
# print(df['minor_axis_length'].describe())


# Filter df where 'minor_axis_length' < 79 and 'filled_area' < 225
filtered_df = df[(df['filled_area'] > 225) & df['filename'].str.contains('limpo')]

# Print summary statistics of filtered_df
print(filtered_df.describe())

import matplotlib.pyplot as plt

# Plot a histogram of the 'filled_area' column in filtered_df
plt.hist(filtered_df['filled_area'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Filled Area')
plt.ylabel('Frequency')
plt.title('Histogram of Filled Area')
plt.grid(True)
plt.show()

