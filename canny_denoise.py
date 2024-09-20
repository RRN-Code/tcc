# -*- coding: utf-8 -*-
"""
Created on Sat May 18 09:32:40 2024

@author: ramos
"""

import cv2
from matplotlib import pyplot as plt
 
path = r"C:\Users\ramos\Documents\Data_Science\coco\train\1\conta01_p0x4.tif"

gray_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Apply median filtering: preserves edges while removing salt-and-pepper noise
median_filtered = cv2.medianBlur(gray_image, 5)

# Apply Gaussian filtering: smooths the overall image without significantly blurring the edges
gaussian_filtered = cv2.GaussianBlur(median_filtered, (5, 5), 0)

# Initial Canny edge detection parameters
low_threshold = 180
high_threshold = 200

# Apply Canny edge detection
edges1 = cv2.Canny(gray_image, low_threshold, high_threshold)
edges2 = cv2.Canny(median_filtered, low_threshold, high_threshold)
edges3 = cv2.Canny(gaussian_filtered, low_threshold, high_threshold)
 
plt.subplot(221),plt.imshow(gray_image,cmap = 'gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(222),plt.imshow(edges1,cmap = 'gray')
plt.title('Median Image')
plt.axis('off')
plt.subplot(223),plt.imshow(edges2,cmap = 'gray')
plt.title('Gaussian Image')
plt.axis('off')
plt.subplot(224),plt.imshow(edges3,cmap = 'gray')
plt.title('Edge Image')
plt.axis('off')
plt.show()
