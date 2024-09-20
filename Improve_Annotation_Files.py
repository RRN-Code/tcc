# -*- coding: utf-8 -*-
"""
Created on Sat May 11 16:59:34 2024

@author: ramos
"""

import os
import numpy as np
from PIL import Image
import cv2

def seg_annotations(input_label, output_label):
    # Ensure the output directory exists
    if not os.path.exists(output_label):
        os.makedirs(output_label)
        
    # Loop through each TIFF file in the input directory
    for filename in os.listdir(input_label):
        if filename.endswith('.tif'):
            # Read the image
            mask_path = os.path.join(input_label, filename)
            # Open the mask image
            mask = Image.open(mask_path)
            
            # Get pixel data as a numpy array
            mask_array = np.array(mask)
            
            # Kernel
            erosion_kernel = np.ones((5, 5), np.uint8)
        
            # Create an empty array to store the eroded masks
            eroded_masks = []
            
            # Loop through each unique element in the mask_array
            for value in np.unique(mask_array):
                # Skip the background (value 0)
                if value == 0:
                    continue
                
                # Create a mask for the current element
                element_mask = (mask_array == value).astype(np.uint8)
                
                # Apply erosion to the mask
                eroded_mask = cv2.erode(element_mask, erosion_kernel, iterations=1)
                
                # Append the eroded mask to the list
                eroded_masks.append(eroded_mask)
                
            # Concatenate all the eroded masks to form the final image
            final_image = np.stack(eroded_masks, axis=-1)
    
            # Sum along the last axis to merge all the eroded masks
            final_image_merged = np.sum(final_image, axis=-1)
            
            # Convert binary_array to uint8 type
            binary_array = final_image_merged.astype(np.uint8)
            
            # Save the modified image with the original filename
            modified_path = os.path.join(output_label, filename)
            # Convert the modified array back to a PIL Image
            modified_image = Image.fromarray(binary_array)
            # Save file
            modified_image.save(modified_path)

            print(f"Image {filename} has been modified.")
