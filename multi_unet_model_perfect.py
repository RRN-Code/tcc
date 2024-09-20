# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:28:00 2024

@author: ramos
"""
import os
import numpy as np
from PIL import Image
from patchify import patchify, unpatchify
# import cv2
# import glob

from keras.utils import normalize
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout


################################################################
def multi_unet_model(IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1, n_classes=3):
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
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

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def get_model():
    return multi_unet_model(IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1, n_classes=3)

model = get_model()

model.load_weights('C:/Users/ramos/Desktop/coco/model_multi_unet.keras')


#################################################


def large_files(input_dir, output_dir, model_name=model):
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
            
            # Patchify the image
            patches = patchify(large_gray_image, (256, 256), step=256)
            
            # Loop for all the patches to predict the Mask
            predicted_patches = []
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    
                    patch = patches[i,j,:,:]
                    patch_norm = np.expand_dims(normalize(np.array(patch), axis=1), 2)
                    patch_input = np.expand_dims(patch_norm, 0)
                    patch_pred = (model.predict(patch_input))
                    patch_pred_img = np.argmax(patch_pred, axis=3)[0,:,:]
                    
                    predicted_patches.append(patch_pred_img)
            
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


# Input and output folders for images
input_folder = 'C:/Users/ramos/Desktop/coco/validation/i_crop'
output_folder = 'C:/Users/ramos/Desktop/coco/validation/m_predicted'


# Predict images
large_files(input_folder, output_folder)


# #################################################


# seg_images = []
# path = 'C:/Users/ramos/Desktop/coco/train/i_crop/*.tif'
# from pathlib import Path
# for file in glob.glob(path):
#     name = Path(file).stem
#     # print(name)
    
#     large_image = cv2.imread(file, 0)
#     patches = patchify(large_image, (256, 256), step=256)
    
#     predicted_patches = []
#     for i in range(patches.shape[0]):
#         for j in range(patches.shape[1]):
            
#             patch = patches[i,j,:,:]
#             patch_norm = np.expand_dims(normalize(np.array(patch), axis=1), 2)
#             patch_input = np.expand_dims(patch_norm, 0)
#             patch_pred = (model.predict(patch_input))
#             patch_pred_img = np.argmax(patch_pred, axis=3)[0,:,:]
            
#             predicted_patches.append(patch_pred_img)
    
#     predicted_patches = np.array(predicted_patches)
#     predicted_patches_reshape = np.reshape(predicted_patches,(patches.shape[0],patches.shape[1], 256, 256))
    
#     unpatched_mask = unpatchify(predicted_patches_reshape, large_image.shape)
    
#     cv2.imwrite('C:/Users/ramos/Desktop/coco/train/seg_mask/'+ name + '.tif', unpatched_mask)
#     print(f'Segmenting image {name} done!')
    
    