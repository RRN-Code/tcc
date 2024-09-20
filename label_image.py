# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 06:40:37 2024

@author: ramos
"""

# import cv2
# import os
# import numpy as np
# from PIL import Image
# # from matplotlib import pyplot as plt

# # Read the image
# image_path = r"C:/Users/ramos/Desktop/coco/train/msks_border_patch_aug/conta01_patch_0_0.jpg"
# image = cv2.imread(image_path)

# # Convert the image to grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# # plt.imshow(gray_image, cmap='gray')
# # plt.title('original Image')
# # plt.show()

# # Create masks for each pixel value range
# mask_below_60 = gray_image < 60
# mask_between_60_and_200 = (gray_image >= 60) & (gray_image <= 200)
# mask_above_200 = gray_image > 200

# # Set pixel values according to masks
# gray_image[mask_below_60] = 0
# gray_image[mask_between_60_and_200] = 1
# gray_image[mask_above_200] = 2

# # plt.imshow(gray_image, cmap='gray')
# # plt.title('Modified Image')
# # plt.show()

# # print(np.unique(gray_image))


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
input_label = 'C:/Users/ramos/Desktop/coco/train/m_crop'
output_label = 'C:/Users/ramos/Desktop/coco/train/m_crop_1x0'

# Label masks
label_images(input_label, output_label)


