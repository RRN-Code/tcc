# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 14:47:41 2024

@author: ramos
"""


import os
import pandas as pd
import numpy as np
from PIL import Image
from skimage import measure


def grain_parameters(input_folder, output_folder):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    global df  # Declare df as a global variable
    
    # Loop through each JPEG file in the input directory
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            # Read the image and turn into array grayscale
            image_path = os.path.join(input_folder, filename)
            image_arr = np.array(Image.open(image_path).convert('L'))


            # Step 4: Label the segmented regions
            labels, num_labels = measure.label(image_arr, connectivity=2, return_num=True)
            
            # regionprops function in skimage measure calculates parameters for each object.
            props = measure.regionprops_table(labels, intensity_image=image_arr, 
                                          properties=['label',
                                                      'area', 'filled_area','convex_area'])
            # Add filename to props dictionary
            props['filename'] = [filename] * num_labels
            
            
            # 'label': The label of each region.
            # 'area': The number of pixels in each region.
            # 'filled_area': The number of pixels in the region, including any holes.
            # 'convex_area': The number of pixels in the convex hull of each region.


            # Append the new row to the DataFrame
            df = pd.concat([df, pd.DataFrame(props)], ignore_index=True)
            print(f'Finish with {filename}')
            
            
    # Specify the full path for the CSV file
    csv_path = 'C:/Users/ramos/Desktop/coco/validation/report/output_clean.csv'

    # Save DataFrame to a CSV file
    df.to_csv(csv_path, index=False)


# Input folders for the clean predicted masks
input_folder = 'C:/Users/ramos/Desktop/coco/validation/m_clean'
output_folder = 'C:/Users/ramos/Desktop/coco/validation/report'

# Create dataframe
df = pd.DataFrame()

# Grains Properties
grain_parameters(input_folder, output_folder)


