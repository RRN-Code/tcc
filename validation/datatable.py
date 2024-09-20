# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 14:47:41 2024

@author: ramos
"""


import os
import pandas as pd
from skimage import measure
from PIL import Image
import numpy as np

base_root = 'C:/Users/ramos/Documents/Data_Science/coco/validation'

# Define the folders containing the TIFF files along with the folder names
folders = [
    ('1', base_root + '/simple_predict_111'),
    ('2', base_root + '/simple_predict_222'),
    ('3', base_root + '/simple_predict_333'),
    ('4', base_root + '/simple_predict_444')
]

# Initialize an empty DataFrame to store the results
df = pd.DataFrame()

# Iterate over each folder and its corresponding name
for folder_name, folder_path in folders:
    # Iterate over each TIFF file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            # Read the image
            image_path = os.path.join(folder_path, filename)
            image_arr = np.array(Image.open(image_path).convert('L'))
            
            # Step 4: Label the segmented regions
            labels, num_labels = measure.label(image_arr, connectivity=2, return_num=True)
            
            # regionprops function in skimage measure calculates parameters for each object.
            props = measure.regionprops_table(labels, intensity_image=image_arr, 
                                          properties=['label',
                                                      'area', 'filled_area','convex_area'])
    
 
            # Add the folder name as a column value
            props['folder_name'] = folder_name
            
            # Add the file name as a column value
            props['file_name'] = filename
            
            # Append the new rows to the DataFrame
            df = pd.concat([df, pd.DataFrame(props)], ignore_index=True)
            
            print(f'Finished processing {filename} in {folder_name}')

    # Specify the full path for the CSV file
    csv_path = base_root + '/' + folder_name + '.csv'
    
    # Save DataFrame to a CSV file
    df.to_csv(csv_path, index=False)


# input_csv1 = base_root + '/1.csv'
# input_csv2 = base_root + '/2.csv'
# input_csv3 = base_root + '/3.csv'
# input_csv4 = base_root + '/4.csv'


# # Read the CSV file into a DataFrame
# df = pd.read_csv(input_csv1)

# # pixel2 to mm2:
# pixel2_to_mm2 = 0.0087


# # Multiply columns 1, 2, and 3 by 0.0087
# df[['area', 'filled_area', 'convex_area']] *= pixel2_to_mm2

# # Assuming df is your DataFrame
# max_df = df.groupby('file_name')['convex_area'].max().reset_index()

# # Print True Positive
# rows = max_df[(max_df['convex_area'] > 600) & max_df['file_name'].str.contains('conta')]
# unique_filenames1 = rows['file_name'].unique()
# print(f' True positive: {len(unique_filenames1)}')

# # Print False Positive
# rows = max_df[(max_df['convex_area'] < 600) & max_df['file_name'].str.contains('conta')]
# unique_filenames2 = rows['file_name'].unique()
# print(f' False positive: {len(unique_filenames2)}')

# # Print False Negative
# rows = max_df[(max_df['convex_area'] > 600) & max_df['file_name'].str.contains('limpo')]
# unique_filenames3 = rows['file_name'].unique()
# print(f' False Negative: {len(unique_filenames3)}')

# # Print True Negative
# rows = max_df[(max_df['convex_area'] < 600) & max_df['file_name'].str.contains('limpo')]
# unique_filenames4 = rows['file_name'].unique()
# print(f' True Negative: {len(unique_filenames4)}')


def calculate_metrics(filename, threshold=600):
    # Read and preprocess CSV files
    input_csv = os.path.join(base_root, f"{filename}.csv")
    df = pd.read_csv(input_csv)
    pixel2_to_mm2 = 0.0087
    df[['area', 'filled_area', 'convex_area']] *= pixel2_to_mm2

    # Group by 'file_name' and find the maximum 'convex_area' for each file
    max_df = df.groupby('file_name')['convex_area'].max().reset_index()

    # Calculate True Positives, False Positives, False Negatives, and True Negatives
    rows_conta = max_df['file_name'].str.contains('conta')
    rows_limpo = max_df['file_name'].str.contains('limpo')
    
    true_positive = len(max_df[(max_df['convex_area'] > threshold) & rows_conta]['file_name'].unique())
    false_negative = len(max_df[(max_df['convex_area'] < threshold) & rows_conta]['file_name'].unique())
    false_positive = len(max_df[(max_df['convex_area'] > threshold) & rows_limpo]['file_name'].unique())
    true_negative = len(max_df[(max_df['convex_area'] < threshold) & rows_limpo]['file_name'].unique())
    
    return true_positive, false_positive, false_negative, true_negative


true_positive, false_positive, false_negative, true_negative = calculate_metrics(4)
print(f"TP: {true_positive} FN: {false_negative}")
print(f"FP: {false_positive} TN: {true_negative}")


