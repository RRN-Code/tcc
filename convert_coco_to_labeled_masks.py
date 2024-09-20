# https://youtu.be/SQng3eIEw-k
"""

Convert coco json labels to labeled masks and copy original images to place 
them along with the masks. 

Dataset from: https://github.com/sartorius-research/LIVECell/tree/main
Note that the dataset comes with: 
Creative Commons Attribution - NonCommercial 4.0 International Public License
In summary, you are good to use it for research purposes but for commercial
use you need to investigate whether trained models using this data must also comply
with this license - it probably does apply to any derivative work so please be mindful. 

You can directly download from the source github page. Links below.

Training json: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json
Validation json: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_val.json
Test json: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_test.json
Images: Download images.zip by following the link: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip

If these links do not work, follow the instructions on their github page. 


"""


import json
import numpy as np
import skimage
import tifffile
import os
from PIL import Image

def create_mask(image_info, annotations, output_folder):
    # Create an empty mask as a numpy array
    mask_np = np.zeros((image_info['height'], image_info['width']), dtype=np.uint16)

    # Counter for the object number
    object_number = 1

    for ann in annotations:
        if ann['image_id'] == image_info['id']:
            # Extract segmentation polygon
            for seg in ann['segmentation']:
                # Convert polygons to a binary mask and add it to the main mask
                rr, cc = skimage.draw.polygon(seg[1::2], seg[0::2], mask_np.shape)
                mask_np[rr, cc] = object_number
                object_number += 1 #We are assigning each object a unique integer value (labeled mask)

    # Save the numpy array as a TIFF using tifffile library
    mask_path = os.path.join(output_folder, image_info['file_name'][:9] + '.tif')
    #  Devido a um erro inicial dos nomes tive que alterar o código de [:7] p/ [:9]
    tifffile.imsave(mask_path, mask_np)

    print(f"Saved mask for {image_info['file_name'][:7]} to {mask_path}")

def main(json_file, mask_output_folder, image_output_folder, original_image_dir):
    # Load COCO JSON annotations
    with open(json_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']

    # Ensure the output directories exist
    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)

    for file in os.listdir(original_image_dir):
        # Check if the file is an image
        if file.endswith(('.jpg', '.jpeg')):
            # Open the image file
            image_path = os.path.join(original_image_dir, file)
            with Image.open(image_path) as img:
                # Construct the output file path with .tif extension
                output_file = file[:7] + '.tif'
                output_path = os.path.join(image_output_folder, output_file)
                
                # Convert and save the image as TIFF
                img.save(output_path, format='TIFF')
                print(f"Converted {file} to {output_file}")

    for img in images:
        # Create the masks
        create_mask(img, annotations, mask_output_folder)

if __name__ == '__main__':
    original_image_dir = 'C:/Users/ramos/Desktop/coco/train'
    json_file = 'C:/Users/ramos/Desktop/coco/train/_annotations.coco.json'
    mask_output_folder = 'C:/Users/ramos/Desktop/coco/train/masks' 
    image_output_folder = 'C:/Users/ramos/Desktop/coco/train/images'
    main(json_file, mask_output_folder, image_output_folder, original_image_dir)



