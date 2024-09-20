# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 06:46:48 2024

@author: ramos
"""

import os
import random
from PIL import Image
from matplotlib import pyplot as plt

# Set the directory paths
original_dir = r"C:\Users\ramos\Documents\Data_Science\coco\validation\i_crop\conta01.tif"
mask_dir = "validation/m_colored"
predicted_dir = "validation/m_watershed"
dirty_dir = "validation/m_clean"
# color_dir = "C:/Users/ramos/Desktop/coco/train/m_pred_clean"

# Get a list of filenames in the directory
filenames = os.listdir(original_dir)

# Set the seed
random.seed(5)

# Select 4 random filenames
random_filenames = random.sample(filenames, 4)

# Plot the images in rows
plt.figure(figsize=(10, 12))

for i, filename in enumerate(random_filenames, 1):
    original_path = os.path.join(original_dir, filename)
    mask_path = os.path.join(mask_dir, filename)
    predicted_path = os.path.join(predicted_dir, filename)
    dirty_path = os.path.join(dirty_dir, filename)
    # color_path = os.path.join(color_dir, filename)

    original = Image.open(original_path)
    mask = Image.open(mask_path)
    predicted = Image.open(predicted_path)
    dirty = Image.open(dirty_path)
    # color = Image.open(color_path)

    plt.subplot(5, 4, i)
    plt.imshow(original)
    plt.title(f'Original Image ({i})'), plt.xticks([]), plt.yticks([])

    plt.subplot(5, 4, i + 4)
    plt.imshow(mask)
    plt.title(f'Predicted mask ({i})'), plt.xticks([]), plt.yticks([])

    plt.subplot(5, 4, i + 8)
    plt.imshow(predicted)
    plt.title(f'Watershed mask ({i})'), plt.xticks([]), plt.yticks([])

    plt.subplot(5, 4, i + 12)
    plt.imshow(dirty)
    plt.title(f'Filtered Grains ({i})'), plt.xticks([]), plt.yticks([])

    # plt.subplot(5, 4, i + 16)
    # plt.imshow(color)
    # plt.title('Grãos Filtrados')

plt.tight_layout()
plt.show()
