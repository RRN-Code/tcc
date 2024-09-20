# -*- coding: utf-8 -*-
"""
Created on Tue May 14 00:50:30 2024

@author: ramos
"""

import os
import cv2
import numpy as np
from PIL import Image
from patchify import patchify, unpatchify
from keras.utils import normalize

def predict_files(input_dir, output_dir, model=None):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Loop through each TIF file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.tif'):
            # Read the image path
            image_path = os.path.join(input_dir, filename)
            
            # Open the image
            large_image = Image.open(image_path)
            
            # Convert the image to grayscale
            grayscale_large = large_image.convert('L')
            
            # Turn the image into a numpy array
            large_gray_image = np.array(grayscale_large)
            
            # Apply median filtering: preserves edges while removing salt-and-pepper noise
            median_filtered = cv2.medianBlur(large_gray_image, 5)

            # Apply Gaussian filtering: smooths the overall image without significantly blurring the edges
            gaussian_filtered = cv2.GaussianBlur(median_filtered, (5, 5), 0)
            
            # Patchify the image
            patches = patchify(gaussian_filtered, (256, 256), step=256)
            
            # Loop for all the patches to predict the Mask
            predicted_patches = []
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    
                    patch = patches[i,j,:,:]
                    patch_norm = np.expand_dims(normalize(np.array(patch), axis=1), 2)
                    patch_input = np.expand_dims(patch_norm, 0)
                    # patch_pred = (model.predict(patch_input))
                    # patch_pred_img = np.argmax(patch_pred, axis=3)[0,:,:]
                    # Predict using the model
                    patch_pred = model.predict(patch_input)
                    # Convert probabilities to binary predictions using thrshold
                    patch_pred_binary = (patch_pred >= 0.5).astype(int)
                    # Assuming you have one class, extract it
                    patch_pred_class = patch_pred_binary[:, :, :, 0]
                    
                    predicted_patches.append(patch_pred_class)
            
            # Unpatchify the Mask
            predicted_patches = np.array(predicted_patches)
            predicted_patches_reshape = np.reshape(predicted_patches,(patches.shape[0],patches.shape[1], 256, 256))
            unpatched_mask = unpatchify(predicted_patches_reshape, large_gray_image.shape)
            unpatched_mask_uint8 = unpatched_mask.astype(np.uint8)
            
            # Save the unpatchify predicted mask
            predicted_filename = f"{os.path.splitext(filename)[0]}.tif"
            predicted_path = os.path.join(output_dir, predicted_filename)
            Image.fromarray(unpatched_mask_uint8).save(predicted_path)
            print(f"Image has been predicted: {filename}")
            
            
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout, add, Activation, multiply

#Define attention block
def attention_block(prev_layer, gated_layer):
    g = Conv2D(filters=gated_layer.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same')(prev_layer)
    x = add([g, gated_layer])
    x = Activation('relu')(x)
    x = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = Activation('sigmoid')(x)
    return multiply([prev_layer, x])


# Define the model
def simple_att_res_unet_model(img_width, img_height, img_channels):
    # Build the model
    inputs = Input((img_width, img_height, img_channels))
    s = inputs

    # Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Attention gate
    attention_gated_c2 = attention_block(p1, c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Attention gate
    attention_gated_c3 = attention_block(p2, c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # Attention gate
    attention_gated_c4 = attention_block(p3, c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)


    # Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, attention_gated_c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = concatenate([c6, c4])  # Residual connection

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, attention_gated_c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c7 = concatenate([c7, c3])  # Residual connection

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, attention_gated_c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    c8 = concatenate([c8, c2])  # Residual connection

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    c9 = concatenate([c9, c1])  # Residual connection

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary()

    return model

# Load the model architecture
model = simple_att_res_unet_model(256, 256, 1)

# Load weights into the model
model.load_weights(r"C:\Users\ramos\Documents\Data_Science\coco\validation\single_unet_4_pre.keras")

input_dir = r"C:\Users\ramos\Documents\Data_Science\coco\validation\i_crop"
output_dir =  r"C:\Users\ramos\Documents\Data_Science\coco\validation\simple_predict_4"

predict_files(input_dir, output_dir, model=model)