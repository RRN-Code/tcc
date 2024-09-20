# https://youtu.be/65qPtD6khzg
"""
Create border pixels from binary masks. 
We can include these border pixels as another class to train a multiclass semantic segmenter
What is the advantage?
We can use border pixels to perform watershed and achieve 'instance' segmentation. 

"""


import cv2
import numpy as np
import glob
from pathlib import Path
import os


# Define function to the border generation
def generate_border(mask, border_size=5, n_erosions=1):
    erosion_kernel = np.ones((9,9), np.uint8)      #Start by eroding edge pixels
    eroded_image = cv2.erode(mask, erosion_kernel, iterations=n_erosions)  

    # Define the kernel size for dilation based on the desired border size (Add 1 to keep it odd)
    kernel_size = 2*border_size + 1 
    dilation_kernel = np.ones((kernel_size, kernel_size), np.uint8)   #Kernel to be used for dilation
    dilated  = cv2.dilate(eroded_image, dilation_kernel, iterations = 1)
    dilated_127 = np.where(dilated == 255, 127, dilated) 	
    original_with_border = np.where(eroded_image > 127, 255, dilated_127)
    
    return original_with_border

# Directory paths
input_dir1 = "C:/Users/ramos/Desktop/coco/train/m"
output_dir1 = "C:/Users/ramos/Desktop/coco/train/msks_ones_zeros"

# Create the output directory if it doesn't exist
os.makedirs(output_dir1, exist_ok=True)

# Get a list of all files in the input directory
file_list = os.listdir(input_dir1)

# Iterate over each file
for file_name in file_list:
    if file_name.endswith('.tif'):  # Check if the file is a JPG file
        # Read the image
        img = cv2.imread(os.path.join(input_dir1, file_name), cv2.IMREAD_GRAYSCALE)
        
        # Change all non-zero pixel values to 1
        img[img != 0] = 255
        
        # Save the modified image to the output directory
        output_path = os.path.join(output_dir1, file_name)
        cv2.imwrite(output_path, img)
        print(f"Processed {file_name} and saved to {output_path}")


# Select the path
input_path2 = "C:/Users/ramos/Desktop/coco/train/msks_ones_zeros/*.jpg"

# Define the output directory
output_dir2 = "C:/Users/ramos/Desktop/coco/train/msks_with_border"

# Create the output directory if it does not exist
os.makedirs(output_dir2, exist_ok=True)

# Iterate over files in the directory
for file in glob.glob(input_path2):
    name = Path(file).stem  # Extract name so processed images can be saved under the same name
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)  # Read each file since we have the full path    
    if img is None:
        print(f"Error: Unable to read {file}")
        continue
    # Process each image
    processed_image = generate_border(img, border_size=5, n_erosions=2)
    # Save images with the same name as the original image/mask
    cv2.imwrite(f"{output_dir2}/{name}.jpg", processed_image)
    print("Finished processing image", name)

   
    
    