#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:14:32 2019

@author: chanontonmai
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import tifffile as tiff

#attack
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, GaussianBlur, RandomBrightnessContrast, Flip, OneOf, Compose, Cutout, Rotate, RandomBrightness,GaussNoise
)
#compose [transform]
cutout = Cutout(p=1, max_h_size=20, max_w_size=20) #cut 8 hole in your wartermarked image
rotate = Rotate(p=1, limit=(20,20)) #rotate 20 degree
brightness = RandomBrightness(p=1, limit=(0.4,0.4)) # change brightness
noise = GaussNoise(p=1,var_limit=(30.0, 30.0)) #add Gauss Noise
blur = GaussianBlur(blur_limit=(1,1),p=1)#Gaussian Blur
trans = [cutout, rotate, brightness, noise,blur]

os.listdir("input/")
#tr = ["cutout","rotate","brightness","noise","blur"]
tr = ["1","2","3","4","5"]
#     print(os.listdir(g_dir))

image_watermarked = tiff.imread('image_with_watermark.tif')
#image_watermarked = image_watermarked*10000
image_watermarked = image_watermarked.astype("float32")
max_image_wm = 255
image_watermarked_norm = image_watermarked/max_image_wm
print(image_watermarked_norm)
for i, tran in enumerate(trans):
    data = {"image": image_watermarked_norm}
                
    augmentation = tran
    augmented = augmentation(**data)
    image = augmented["image"]
    image = image
    new_image_name =str(tr[i]) + '_' + 'image_with_watermark' + '.tif'
    tiff.imwrite(new_image_name,image)