# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:52:21 2024

@author: ramos
"""

import os
import numpy as np
import cv2


def watershed_files(input_folder, output_folder):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        

    # Loop through each tif file in the input directory
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            
            # Find the file path
            mask_path = os.path.join(input_folder, filename)
            
            # Read the mask
            img = cv2.imread(mask_path)
            
            # Turn the mask between 0 and 255
            img[img==2] = 0
            img[img==1] = 255
            
            # Turn the 3 channel watershed image into 1 channel
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            # Remove some noise
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel, iterations = 5)
            
            # Define sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)
            
            # Define sure foreground area
            percentage = 0.10
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret2, sure_fg = cv2.threshold(dist_transform,percentage*dist_transform.max(),255,0)
             
            # Change sure_fg type to uint8
            sure_fg = np.uint8(sure_fg)
            
            # Define unknown area
            unknown = cv2.subtract(sure_bg,sure_fg)
            
            # Define markers to watershed Operation
            ret3, markers = cv2.connectedComponents(sure_fg)
             
            # Add one to all labels so that sure background is not 0, but 1
            markers = markers+1
             
            # Now, mark the region of unknown with zero
            markers[unknown==255] = 0
            
            # Performe watershed
            markers = cv2.watershed(img,markers)
            
            # Color black the line of watershed in the original image
            img[markers == -1] = [0,0,0]
            
            # Crop original image to remove border contour line
            cropped_img = img[1:-1, 1:-1]
            
            # Turn the 3 channel watershed image into 1 channel
            cropped_gray = cv2.cvtColor(cropped_img,cv2.COLOR_BGR2GRAY)
            
            # Improve the thickness of the watershed line
            cropped_eroded = cv2.erode(cropped_gray,kernel,iterations=1)
            
            # Define the components of the new image
            ret4, cropped_markers = cv2.connectedComponents(cropped_eroded)
            
            # Create false color image with black background and colored objects
            colors = np.random.randint(0, 255, size=(ret4, 3), dtype=np.uint8)
            colors[0] = [0, 0, 0]  # black background
            false_colors = colors[cropped_markers]

            # Save the patch
            watershed_filename = f"{os.path.splitext(filename)[0]}.tif"
            output_path = os.path.join(output_folder, watershed_filename)
            cv2.imwrite(output_path, false_colors)
            print(f"Watershed was applied to {filename}.")

            

# Input and output folders for images
input_folder = "C:/Users/ramos/Desktop/coco/validation/m_predicted"
output_folder = 'C:/Users/ramos/Desktop/coco/validation/m_watershed'


# Patchify images
watershed_files(input_folder, output_folder)
