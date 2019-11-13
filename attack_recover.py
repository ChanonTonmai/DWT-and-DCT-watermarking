#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:10:46 2019

@author: chanontonmai
"""
from dwt_wtm import *
import tifffile as tiff

model = 'haar'
level = 2
wm_size = 64

print('Prepare Attack Check')
image_test_1 = 'attacked/1_image_with_watermark.tif'
image_test_2 = 'attacked/2_image_with_watermark.tif'
image_test_3 = 'attacked/3_image_with_watermark.tif'
image_test_4 = 'attacked/4_image_with_watermark.tif'
image_test_5 = 'attacked/5_image_with_watermark.tif'


image_array_rgb_1 = tiff.imread(image_test_1)
image_array_rgb_2 = tiff.imread(image_test_2)
image_array_rgb_3 = tiff.imread(image_test_3)
image_array_rgb_4 = tiff.imread(image_test_4)
image_array_rgb_5 = tiff.imread(image_test_5)

image_array_rgb_1 = image_array_rgb_1 *255
image_array_rgb_2 = image_array_rgb_2 *255
image_array_rgb_3 = image_array_rgb_3 *255
image_array_rgb_4 = image_array_rgb_4 *255
image_array_rgb_5 = image_array_rgb_5 *255

#img_test_1, image_array, image_array_rgb_1 = image_resize(image_test_1, 2048)
#img_test_2, image_array, image_array_rgb_2 = image_resize(image_test_2, 2048)
#img_test_3, image_array, image_array_rgb_3 = image_resize(image_test_3, 2048)
#img_test_4, image_array, image_array_rgb_4 = image_resize(image_test_4, 2048)
#img_test_5, image_array, image_array_rgb_5 = image_resize(image_test_5, 2048)


recover_watermark(image_array = image_array_rgb_1[:,:,2], model=model, level = level,wm_size=wm_size, name = '_test0')
recover_watermark(image_array = image_array_rgb_2[:,:,2], model=model, level = level,wm_size=wm_size, name = '_test1')
recover_watermark(image_array = image_array_rgb_3[:,:,2], model=model, level = level,wm_size=wm_size, name = '_test2')
recover_watermark(image_array = image_array_rgb_4[:,:,2], model=model, level = level,wm_size=wm_size, name = '_test3')
recover_watermark(image_array = image_array_rgb_5[:,:,2], model=model, level = level,wm_size=wm_size, name = '_test4')
print('Finish')