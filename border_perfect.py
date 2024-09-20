# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:36:52 2024

@author: ramos
"""

import os
import numpy as np
from PIL import Image
import cv2


# Define function to the border generation
def generate_border(mask_array, border_size= 12, n_erosions= 1):
    erosion_kernel = np.ones((7,7), np.uint8)      #Start by eroding edge pixels
    eroded_mask = cv2.erode(mask_array, erosion_kernel, iterations= n_erosions) 

    # Define the kernel size for dilation based on the desired border size (Add 1 to keep it odd)
    kernel_size = 2*border_size + 1 
    dilation_kernel = np.ones((kernel_size, kernel_size), np.uint8)   #Kernel to be used for dilation
    dilated_mask  = cv2.dilate(eroded_mask, dilation_kernel, iterations = 1)
    dilated_2 = np.where(dilated_mask == 1, 2, dilated_mask) 	
    original_with_border = np.where(eroded_mask > 0, 1, dilated_2)
    
    return original_with_border

# Define function to the border files creation
def border_files(input_label, output_label):
    # Ensure the output directory exists
    if not os.path.exists(output_label):
        os.makedirs(output_label)
        
    # Loop through each TIFF file in the input directory
    for filename in os.listdir(input_label):
        if filename.endswith('.tif'):
            # Read the mask
            mask_path = os.path.join(input_label, filename)
            # Open the mask
            mask = Image.open(mask_path)
            
            # Get pixel data as a numpy array
            mask_array = np.array(mask)
        
            # Process each mask
            processed_image = generate_border(mask_array, border_size= 12, n_erosions= 1)
            
            # Convert the numpy array back to a PIL Image
            modified_mask = Image.fromarray(processed_image.astype(np.uint8))  # Ensure data type is consistent
            
            unique_values = np.unique(processed_image)
            
            # Save the modified mask with the original filename
            modified_path = os.path.join(output_label, filename)
            modified_mask.save(modified_path)  # Use PIL's save() method instead of cv2.imwrite()

            print(f"Image {filename} has been modified. The unique values are {unique_values}")


# Input and output folders for images
input_label = "C:/Users/ramos/Documents/Data_Science/coco/train/masks_improved"
output_label = "C:/Users/ramos/Documents/Data_Science/coco/train/m_border_improved"

# Label masks
border_files(input_label, output_label)