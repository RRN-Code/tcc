# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 21:36:58 2024

@author: ramos
"""

import os
import shutil

def check_and_copy_files(input_label, output_label, keyword):
    # Ensure the output directory exists
    if not os.path.exists(output_label):
        os.makedirs(output_label)

    # List files in the specified folder
    files = os.listdir(input_label)

    # Iterate through files
    for file in files:
        # Check if 'aug' is present in the filename
        if keyword in file:
            print(f"'{file}' contains '{keyword}'.")
        else:
            # Copy the file to the output directory
            source_path = os.path.join(input_label, file)
            destiny_path = os.path.join(output_label, file)
            shutil.copy(source_path, destiny_path)
            print(f"'{file}' doesn't contain '{keyword}'.")

# Example usage
input_label = 'C:/Users/ramos/Documents/Data_Science/coco/train/m_multi_dataset'
output_label = 'C:/Users/ramos/Documents/Data_Science/coco/train/m_short'
keyword ='aug'
check_and_copy_files(input_label, output_label, keyword)

