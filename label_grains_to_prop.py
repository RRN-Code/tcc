# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 06:40:53 2024

@author: ramos
"""
import os
import cv2
import numpy as np
from PIL import Image
from skimage.segmentation import clear_border


def grain_cleaner(input_dir, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Loop through each JPEG file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.tif'):
            # Read the image and turn into array grayscale
            image_path = os.path.join(input_dir, filename)
            image_arr = np.array(Image.open(image_path).convert('L'))

            # Turn border pixels to background pixels
            image_arr[image_arr == 2] = 0

            # Remove the grains that touch the file limits
            image_arr = clear_border(image_arr)

            # Label every grain in the file
            ret2, markers = cv2.connectedComponents(image_arr, connectivity=4)
            markers = markers.astype(np.uint8)
            
            # Find contours of objects
            contours, _ = cv2.findContours(markers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Iterate through contours and remove small ones
            area_minima = 20_000
            
            # Iterate through contours and remove shapeless ones
            area_ratio = 1.1         
            
            for contour in contours:
                area = cv2.contourArea(contour)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if area < area_minima or (hull_area / area) >= area_ratio:
                    cv2.drawContours(markers, [contour], -1, 0, -1)  # Fill contour with black color

            
            # Create false color image with black background and colored objects
            colors = np.random.randint(0, 255, size=(ret2, 3), dtype=np.uint8)
            colors[0] = [0, 0, 0]  # black background
            false_colors = colors[markers]

            
            # Save the thresh
            thresh_filename = f"{os.path.splitext(filename)[0]}.tif"
            thresh_path = os.path.join(output_dir, thresh_filename)
            Image.fromarray(false_colors).save(thresh_path)
            print(f"Image {filename} has been false colored.")

                      

# Input and output folders for images
input_dir = 'C:/Users/ramos/Desktop/coco/train/m_predicted'
output_dir = 'C:/Users/ramos/Desktop/coco/train/m_pred_clean3'


# Clean masks grains
grain_cleaner(input_dir, output_dir)