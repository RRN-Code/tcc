# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:41:01 2024

@author: ramos
"""

import os
import cv2
import numpy as np
from skimage.segmentation import clear_border

# Set the seed
np.random.seed(42)
 
def grain_cleaner(input_dir, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Loop through each JPEG file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.tif'):    
            # Read the path
            image_path = os.path.join(input_dir, filename)
            # Read the mask
            img = cv2.imread(image_path)
             
            # Turn the 3 channel watershed image into 1 channel
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            # Remove the grains that touch the file limits
            img_clear = clear_border(gray)
            
            # Label every grain in the file
            ret, markers = cv2.connectedComponents(img_clear, connectivity=4)
            markers = markers.astype(np.uint8)
            
            # Find contours of objects
            contours, _ = cv2.findContours(markers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Iterate through contours and remove small ones
            area_minima = 20_000
            
            # Iterate through contours and remove shapeless ones
            hull_area_ratio = 1.20
            
            for contour in contours:
                area = cv2.contourArea(contour)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if area < area_minima or (hull_area / area) >= hull_area_ratio:
                    cv2.drawContours(markers, [contour], -1, 0, -1)  # Fill contour with black color
            
            
            # Create false color image with black background and colored objects
            colors = np.random.randint(0, 255, size=(ret, 3), dtype=np.uint8)
            colors[0] = [0, 0, 0]  # black background
            false_colors = colors[markers]
            
            
            # Save the thresh
            clean_filename = f"{os.path.splitext(filename)[0]}.tif"
            output_path = os.path.join(output_dir, clean_filename)
            cv2.imwrite(output_path, false_colors)
            print(f"Image {filename} has been colored.")


# Input and output folders for images
input_dir = 'C:/Users/ramos/Desktop/coco/validation/m_watershed'
output_dir = 'C:/Users/ramos/Desktop/coco/validation/m_clean'

# Clean masks grains
grain_cleaner(input_dir, output_dir)