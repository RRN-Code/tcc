# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 21:08:33 2024

@author: ramos
"""

import os

def copy_files(src_folders, dest_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # Iterate over all source folders
    for src_folder in src_folders:
        # Iterate over all files in the current source folder
        for file_name in os.listdir(src_folder):
            # Build the full path for both source and destination files
            src_file = os.path.join(src_folder, file_name)
            dest_file = os.path.join(dest_folder, file_name)
            
            # Open the source file for reading
            with open(src_file, 'rb') as src:
                # Open or create the destination file for writing
                with open(dest_file, 'wb') as dest:
                    # Read from the source file and write to the destination file
                    dest.write(src.read())
                    
            print(f"Copied {src_file} to {dest_file}")

# Example usage:
source_folders1 = ['C:/Users/ramos/Desktop/coco/train/imgs_patch',
                  'C:/Users/ramos/Desktop/coco/train/imgs_patch_augmented']

destination_folder1 = 'C:/Users/ramos/Desktop/coco/train/imgs_dataset'

source_folders2 = ['C:/Users/ramos/Desktop/coco/train/msks_patch',
                  'C:/Users/ramos/Desktop/coco/train/msks_patch_augmented']

destination_folder2 = 'C:/Users/ramos/Desktop/coco/train/msks_dataset'


copy_files(source_folders1, destination_folder1)

copy_files(source_folders2, destination_folder2)
