# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 20:31:55 2024

@author: ramos
"""

import os
import numpy as np
from PIL import Image

def patchify_files(input_dir, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Loop through each JPEG file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.tif'):
            # Read the image
            image_path = os.path.join(input_dir, filename)
            image = np.array(Image.open(image_path))

            # Define patch size and step
            patch_size = 256
            step = 256

            # Crop the image into patches
            num_rows = (image.shape[0] - patch_size) // step + 1
            num_cols = (image.shape[1] - patch_size) // step + 1

            for i in range(num_rows):
                for j in range(num_cols):
                    # Calculate patch coordinates
                    start_x = i * step
                    start_y = j * step
                    end_x = start_x + patch_size
                    end_y = start_y + patch_size

                    # Crop the patch
                    patch = image[start_x:end_x, start_y:end_y]

                    # Save the patch
                    patch_filename = f"{os.path.splitext(filename)[0]}_p{i}x{j}.tif"
                    patch_path = os.path.join(output_dir, patch_filename)
                    Image.fromarray(patch).save(patch_path)

            print(f"Image {filename} has been patchified.")

