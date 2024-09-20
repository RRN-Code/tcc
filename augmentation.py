# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 20:46:28 2024

@author: ramos
"""

import os
import numpy as np
from PIL import Image
import albumentations as A

def augment_images(folder):
    
    # Check if the folder exists
    if not os.path.exists(folder):
        raise ValueError(f"The specified folder '{folder}' does not exist.")
    
    # Define augmentation pipeline
    augmentation = A.Compose([
        A.HorizontalFlip(p=1.0),  # Flip horizontally with probability 1.0
        A.VerticalFlip(p=1.0)  # Flip horizontally with probability 1.0
    ])
    
    # Loop through each JPEG file in the input directory
    for filename in os.listdir(folder):
        if filename.endswith('.tif'):
            # Read the image
            image_path = os.path.join(folder, filename)
            image = np.array(Image.open(image_path))

            # Apply augmentation
            augmented_image = augmentation(image=image)['image']

            # Save the augmented image
            augmented_filename = f"{os.path.splitext(filename)[0]}_aug.tif"
            augmented_path = os.path.join(folder, augmented_filename)
            Image.fromarray(augmented_image).save(augmented_path)

            print(f"Image {filename} has been augmented.")


