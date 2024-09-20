# -*- coding: utf-8 -*-
"""
Created on Tue May 14 01:26:28 2024

@author: ramos
"""

from models_simple import simple_unet_model, simple_res_unet_model, simple_att_unet_model, simple_att_res_unet_model
from predict import predict_files

base_root = r'C:\Users\ramos\Documents\Data_Science\coco\validation'

# Define input-output pairs
input_output_pairs = [
    ( base_root +'/i_crop', base_root +'/simple_unet', simple_unet_model(256, 256, 1), base_root +'/single_unet_1_pre.keras'),
    ( base_root +'/i_crop', base_root +'/simple_res', simple_res_unet_model(256, 256, 1), base_root +'/single_unet_2_pre.keras'),
    ( base_root +'/i_crop', base_root +'/simple_att', simple_att_unet_model(256, 256, 1), base_root +'/single_unet_3_pre.keras'),
    ( base_root +'/i_crop', base_root +'/simple_att_res', simple_att_res_unet_model(256, 256, 1), base_root +'/single_unet_4_pre.keras')
]

# Load models and predict images
for input_dir, output_dir, model, weights_path in input_output_pairs:
    model.load_weights(weights_path)
    predict_files(input_dir, output_dir, model)