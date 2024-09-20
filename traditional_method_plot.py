# -*- coding: utf-8 -*-
"""
Created on Sun May 26 12:58:47 2024

@author: ramos
"""



import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

path1 = r"C:\Users\ramos\Documents\Data_Science\coco\train\i_crop\conta02.tif"
path2 = r"C:\Users\ramos\Documents\Data_Science\coco\train\m_crop_single_flipped\conta02.tif"

# Load the image
image = Image.open(path1)
mask = Image.open(path2).convert('L')

gray_image = image.convert('L')

gray_image = np.array(gray_image)

# Binarization using a threshold
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Apply median filtering to eliminate salt and pepper noise
median_filtered_image = cv2.medianBlur(binary_image, 5)

# Apply Canny edge detection
edges = cv2.Canny(median_filtered_image, 100, 200)

# Invert the edges
inverted_edges = cv2.bitwise_not(edges)

# # Perform dilation to thicken edges
kernel = np.ones((5, 5), np.uint8)
dilated_image = cv2.dilate(edges, kernel, iterations=1)

# Perform closing operation to connect edges
closed_image = cv2.morphologyEx(dilated_image, cv2.MORPH_CLOSE, kernel)

# Invert the edges
inverted_edges = cv2.bitwise_not(closed_image)

# Find connected components
num_labels, labels = cv2.connectedComponents(inverted_edges)


# Create false color image with black background and colored objects
colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
colors[0] = [0, 0, 0]  # black background
false_colors = colors[labels]

# # Map component labels to hue values
# label_hue = np.uint8(179 * labels / np.max(labels))
# blank_ch = 255 * np.ones_like(label_hue)
# colored_image = cv2.merge([label_hue, blank_ch, blank_ch])

# # Convert to BGR for display
# colored_image = cv2.cvtColor(colored_image, cv2.COLOR_HSV2BGR)

# # Set background label to black
# colored_image[label_hue == 0] = 0

# Display the results
titles = ['Original Image', 'Grayscale Image', 'Binary Image', 'Median Filtered Image', 'Canny Edges', 'Closed Image', 'Colored Components', 'Original Annotation']
images = [image, gray_image, binary_image, median_filtered_image, edges, closed_image, false_colors, mask]

# # Display the results
# titles = ['Original Image', 'Canny Edges',  'Colored Components']
# images = [image, closed_image, colored_image]

plt.figure(figsize=(15, 7))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.tight_layout()    
plt.show()

