# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 06:40:37 2024

@author: ramos
"""


import os
import numpy as np
from PIL import Image

def label_images(input_label, output_label):
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

            # Convert the image to grayscale to handle multichannel images
            mask = mask.convert('L')
            
            # Get pixel data as a numpy array
            mask_array = np.array(mask)
        
            # Set all non-zero values to 1
            mask_array[mask_array != 0] = 1
            
            # Convert the numpy array back to a PIL Image
            modified_mask = Image.fromarray(mask_array.astype(np.uint8))  # Ensure data type is consistent
            
            unique_values = np.unique(mask_array)
            
            # Save the modified image with the original filename
            modified_path = os.path.join(output_label, filename)
            modified_mask.save(modified_path)  # Use PIL's save() method instead of cv2.imwrite()

            print(f"Image {filename} has been modified. The unique values are {unique_values}")


# Input and output folders for images
input_label = 'C:/Users/ramos/Desktop/coco/train/m_crop_1x0'
output_label = 'C:/Users/ramos/Desktop/coco/train/m_border'

# Label masks
label_images(input_label, output_label)


