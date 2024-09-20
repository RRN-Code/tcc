# -*- coding: utf-8 -*-
"""
Created on Tue May 14 01:50:01 2024

@author: ramos
"""

from watershed import watershed_files
from final_grains import clean_files

base_root = 'C:/Users/ramos/Documents/Data_Science/coco/validation'

# def process_images(input_output_pairs, processing_function):
    
#     for input_dir, output_dir in input_output_pairs:
#         processing_function(input_dir, output_dir)


# input_output_pairs1 = [
#     (base_root + '/simple_predict_1', base_root + '/simple_predict_11'),
#     (base_root + '/simple_predict_2', base_root + '/simple_predict_22'),
#     (base_root + '/simple_predict_3', base_root + '/simple_predict_33'),
#     (base_root + '/simple_predict_4', base_root + '/simple_predict_44')
# ]

# input_output_pairs2 = [
#     (base_root + '/simple_predict_11', base_root + '/simple_predict_111'),
#     (base_root + '/simple_predict_22', base_root + '/simple_predict_222'),
#     (base_root + '/simple_predict_33', base_root + '/simple_predict_333'),
#     (base_root + '/simple_predict_44', base_root + '/simple_predict_444')
# ]


# process_images(input_output_pairs1, watershed_files)
# process_images(input_output_pairs2, clean_files)


def process_images(input_output_pairs, processing_function):
    
    for input_dir, output_dir in input_output_pairs:
        processing_function(input_dir, output_dir)


input_output_pairs1 = [
    (base_root + '/canny', base_root + '/canny1')
]

input_output_pairs2 = [
    (base_root + '/canny1', base_root + '/canny11')
]


process_images(input_output_pairs1, watershed_files)
process_images(input_output_pairs2, clean_files)


