#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:11:20 2019

@author: chanontonmai
"""
import numpy as np
import math
import pywt
import os
import cv2
from PIL import Image
from scipy.fftpack import dct
from scipy.fftpack import idct
from matplotlib import pyplot as plt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis

import tifffile as tiff

# In[3]:


def image_resize(image_name, size):
    img = Image.open(image_name).resize((size, size), 1)
    img_gray = img.convert('L')
    img_rgb = img.convert('RGB')
    image_name2 = 'rgb_'+str(image_name)
    img_gray.save('./dataset/' + image_name)
    img_rgb.save('./dataset/' + image_name2)

 
    image_array = np.array(img_gray.getdata(), dtype=np.float).reshape((size, size))
    image_array_rgb = np.array(img_rgb.getdata(), dtype=np.float).reshape((size, size,3))    

    return img, image_array, image_array_rgb


# In[4]:


def mul_wavelet_dec(imArray):
    shape = imArray.shape
    fig, axes = plt.subplots(2, 4, figsize=[14, 8])
    max_lev = 3       # how many levels of decomposition to draw
    label_levels = 3  # how many levels to explicitly label on the plots
    for level in range(0, max_lev + 1):
        if level == 0:
            # show the original image before decomposition
            axes[0, 0].set_axis_off()
            axes[1, 0].imshow(imArray, cmap=plt.cm.gray)
            axes[1, 0].set_title('Image')
            axes[1, 0].set_axis_off()
            continue

        # plot subband boundaries of a standard DWT basis
        draw_2d_wp_basis(shape, wavedec2_keys(level), ax=axes[0, level],
                         label_levels=label_levels)
        axes[0, level].set_title('{} level\ndecomposition'.format(level))

        # compute the 2D DWT
        c = pywt.wavedec2(imArray, 'haar', mode='periodization', level=level)
        # normalize each coefficient array independently for better visibility
        c[0] /= np.abs(c[0]).max()
        for detail_level in range(level):
            c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]
        # show the normalized coefficients
        arr, slices = pywt.coeffs_to_array(c)
        axes[1, level].imshow(arr, cmap=plt.cm.gray)
        axes[1, level].set_title('Coefficients\n({} level)'.format(level))
        axes[1, level].set_axis_off()


# In[5]:


def process_coefficients(imArray, model, level):
    
    coeffs=pywt.wavedec2(data = imArray, wavelet = model, level = level)
    coeffs_H=list(coeffs) 
   
    return coeffs_H


# In[6]:


def apply_dct(image_array):
    size = image_array[0].__len__()
    all_subdct = np.empty((size, size))
    for i in range (0, size, 8):
        for j in range (0, size, 8):
            subpixels = image_array[i:i+8, j:j+8]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
            all_subdct[i:i+8, j:j+8] = subdct

    return all_subdct


# In[7]:


def inverse_dct(all_subdct):
    size = all_subdct[0].__len__()
    all_subidct = np.empty((size, size))
    for i in range (0, size, 8):
        for j in range (0, size, 8):
            subidct = idct(idct(all_subdct[i:i+8, j:j+8].T, norm="ortho").T, norm="ortho")
            all_subidct[i:i+8, j:j+8] = subidct

    return all_subidct


# In[8]:


def embed_watermark(watermark_array, orig_image):
    orig_image_dummy = orig_image
    watermark_flat = watermark_array.ravel()
    size = orig_image.__len__()
    ind = 0
    
    for x in range (0, size, 8):
        for y in range (0, size, 8):
            if ind < watermark_flat.__len__():
                subdct = orig_image[x:x+8, y:y+8]
                subdct[5][5] = watermark_flat[ind]
                orig_image[x:x+8, y:y+8] = subdct
                ind += 1 


    return orig_image


# In[9]:


def print_image_from_array(image_array, name):
  
    image_array_copy = image_array.clip(0, 255)
    image_array_float = image_array_copy.astype("float32")
    image_array_copy = image_array_copy.astype("uint8")
    img = Image.fromarray(image_array_copy)
    img.save('./result/' + name + '.bmp')
    tiff.imsave('image_with_watermark.tif', image_array_float)
    return img


# In[10]:


def get_watermark(dct_watermarked_coeff, watermark_size):
    
    subwatermarks = []

    for x in range (0, dct_watermarked_coeff.__len__(), 8):
        for y in range (0, dct_watermarked_coeff.__len__(), 8):
            coeff_slice = dct_watermarked_coeff[x:x+8, y:y+8]
            subwatermarks.append(coeff_slice[5][5])
            
    watermark = np.array(subwatermarks).reshape(watermark_size, watermark_size)

    return watermark

def recover_watermark(image_array, model, level, wm_size, name):


    coeffs_watermarked_image = process_coefficients(image_array, model, level=level)
    #coeffs_watermarked_image_l2 = process_coefficients(coeffs_watermarked_image[1][0], model, level=level)
    dct_watermarked_coeff = apply_dct(coeffs_watermarked_image[0])
    
    watermark_array = get_watermark(dct_watermarked_coeff, wm_size)*64

    watermark_array =  np.uint8(watermark_array)

    #Save result
    img = Image.fromarray(watermark_array)
    name = './result/recovered_watermark'+name+'.bmp'
    img.save(name)