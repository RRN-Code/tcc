# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:28:15 2024

@author: ramos
"""

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2

mask_path = "C:/Users/ramos/Documents/Data_Science/coco/train/masks/conta01.tif"

# Open Original image
mask = Image.open(mask_path)

#Plot Original Image
plt.imshow(mask)
plt.show()

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

#Plot final result
plt.imshow(final_image_merged)
plt.show()

# Convert binary_array to uint8 type
binary_array = final_image_merged.astype(np.uint8)

# Find connected components in the binary array
num_components, labels = cv2.connectedComponents(binary_array)

# Subtract 1 from num_components to exclude the background component (0)
num_elements = num_components - 1

print("Number of connected elements:", num_elements)

