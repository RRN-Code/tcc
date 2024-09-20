# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 06:50:18 2024

@author: ramos
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

def adjust_image_properties(image, saturation=1.0, brightness=0.0, contrast=1.0, exposure=1.0, shadows=0.0):
    # Convert the image to float32 format
    image = image.astype(np.float32) / 255.0
    
    # Adjust saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] *= saturation
    
    # Adjust brightness
    image += brightness
    
    # Adjust contrast
    image = (image - 0.5) * contrast + 0.5
    
    # Adjust exposure
    image *= exposure
    
    # Adjust shadows
    image += shadows
    
    # Clip the pixel values to ensure they are within the valid range [0, 1]
    image = np.clip(image, 0, 1)
    
    # Convert the image back to uint8 format
    image = (image * 255).astype(np.uint8)
    
    # Convert the images back to BGR color space
    adjusted_image = cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB)
    
    return adjusted_image

image_path = 'C:/Users/ramos/Desktop/coco/train/imgs/conta01.jpg'

# Load an image
image = cv2.imread(image_path)

# Adjust image properties
adjusted_image = adjust_image_properties(image, saturation=2, brightness=10, contrast=10, exposure=10, shadows=10)


plt.imshow(image)
plt.title('Original Image')
plt.show()

plt.imshow(adjusted_image)
plt.title('Adjusted Image')
plt.show()
