# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:26:47 2024

@author: ramos
"""

import os
import cv2
import numpy as np
from PIL import Image
from skimage.segmentation import clear_border
from matplotlib import pyplot as plt

# There is a Crop layer that the HED network, without the crop layer,
# the final result will be shifted to the right and bottom cropping the image
class CropLayer(object):
    def __init__(self, params, blobs):
        # initialize our starting and ending (x, y)-coordinates of the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        # the crop layer will receive two inputs -- we need to crop
        # the first input blob to match the shape of the second one,
        # keeping the batch size and number of channels
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        # return the shape of the volume: the actual crop during the forward pass
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        # use the derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY,
                self.startX:self.endX]]


def hed_processor(input_folder, output_folder):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Loop through each JPEG file in the input directory
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            (H, W) = img.shape[:2]


            # construct a blob out of the input image 
            mean_pixel_values= np.average(img, axis = (0,1))
            blob = cv2.dnn.blobFromImage(img, scalefactor=0.3, size=(W, H),
                                         mean=(mean_pixel_values[0], mean_pixel_values[1], mean_pixel_values[2]),
                                         swapRB= False, crop=False)


            # set the blob as the input to to compute the edges
            net.setInput(blob)
            hed = net.forward()
            hed = hed[0,0,:,:]  #Drop the other axes 
            hed = (255 * hed).astype("uint8")  #rescale to 0-255


            # Load segmented binary image, Gaussian blur, grayscale, Otsu's threshold
            blur = cv2.GaussianBlur(hed, (3,3), 0)

            # Thresh
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            # Perform connected component labeling
            n_labels, labels = cv2.connectedComponents(thresh, connectivity=4)
            
            # Find contours of objects
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Iterate through contours and remove small ones
            area_minima = 20_000
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < area_minima:
                    cv2.drawContours(labels, [contour], -1, 0, -1)  # Fill contour with black color
            
            # Remove the grains that touch the file limits
            labels_clean = clear_border(labels)


            # Convert to uint8
            labels_clean = np.uint8(labels_clean)
            
            # Create false color image with black background and colored objects
            colors = np.random.randint(0, 255, size=(n_labels, 3), dtype=np.uint8)
            colors[0] = [0, 0, 0]  # black background
            false_colors = colors[labels_clean]

            
            # Save the thresh
            thresh_filename = f"{os.path.splitext(filename)[0]}.tif"
            thresh_path = os.path.join(output_folder, thresh_filename)
            Image.fromarray(false_colors).save(thresh_path)
            plt.imshow(false_colors)
            print(f"Image {filename} has been thresholded.")




# The pre-trained model that OpenCV uses has been trained in Caffe framework
protoPath = 'C:/Users/ramos/Desktop/coco/deploy.prototxt'
modelPath = 'C:/Users/ramos/Desktop/coco/hed_pretrained_bsds.caffemodel'
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


# register our crop layer with the model
cv2.dnn_registerLayer("Crop", CropLayer)


# Input and output folders for images
input_folder = 'C:/Users/ramos/Desktop/coco/train/i_crop'
output_folder = 'C:/Users/ramos/Desktop/coco/train/m_hed_thresh_3'


# Patchify images
hed_processor(input_folder, output_folder)