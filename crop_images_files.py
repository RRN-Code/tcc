# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 15:22:18 2024

@author: ramos
"""
        
import os
from PIL import Image


def create_output_folder(output_folder):
    """Create output folder if it doesn't exist."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


def crop_files(input_folder, output_folder, x, y, width, height, input_extension='.tif', output_extension='.tif'):
    """Crop images/masks from input folder and save to output folder."""
    create_output_folder(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(input_extension):
            input_path = os.path.join(input_folder, file_name)
            output_file_name = file_name[:7] + output_extension
            output_path = os.path.join(output_folder, output_file_name)
            
            # Open image/mask using PIL
            img = Image.open(input_path)
            
            # Crop the image/mask
            cropped_img = img.crop((x, y, x+width, y+height))
            
            # Save the cropped image/mask
            cropped_img.save(output_path)
            
            print(f"Cropped {file_name} and saved to {output_path}")

        
        
        
        
        