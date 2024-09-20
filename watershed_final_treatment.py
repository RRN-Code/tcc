# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 06:40:08 2024

@author: ramos
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


dirty_path = "train/masks/conta01.tif"

# Read image
img = cv.imread(dirty_path)

plt.imshow(img)
plt.title('Original Image')
plt.show()

# Turn the colorful image into Black&White
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        # Check if the pixel is not [0, 0, 0]
        if not all(img[i, j] == [0, 0, 0]):
            # Set the pixel to [255, 255, 255]
            img[i, j] = [255, 255, 255]

# Turn the 3 channel image into 1 channel
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Threshold not necessary
ret1, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# Remove some noise
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 5)

# Define sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)

# Define sure foreground area
percentage = 0.10
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret2, sure_fg = cv.threshold(dist_transform,percentage*dist_transform.max(),255,0)
 
# Change sure_fg type to uint8
sure_fg = np.uint8(sure_fg)

# Define unknown area
unknown = cv.subtract(sure_bg,sure_fg)

# Define markers to watershed Operation
ret3, markers = cv.connectedComponents(sure_fg)
 
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
 
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

# Performe watershed
markers = cv.watershed(img,markers)

# Color black the line of watershed in the original image
img[markers == -1] = [0,0,0]


# Crop original image to remove border contour line
cropped_img = img[1:-1, 1:-1]

# Turn the 3 channel watershed image into 1 channel
cropped_gray = cv.cvtColor(cropped_img,cv.COLOR_BGR2GRAY)

# Improve the thickness of the watershed line
cropped_eroded = cv.erode(cropped_gray,kernel,iterations=1)

# Define the components of the new image
ret4, cropped_markers = cv.connectedComponents(cropped_eroded)

# Create false color image with black background and colored objects
colors = np.random.randint(0, 255, size=(ret4, 3), dtype=np.uint8)
colors[0] = [0, 0, 0]  # black background
false_colors = colors[cropped_markers]

plt.imshow(false_colors)
plt.title('Watershed Image')
plt.show()

