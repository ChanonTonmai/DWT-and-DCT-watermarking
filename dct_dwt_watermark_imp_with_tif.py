# coding: utf-8

# In[1]:


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

from dwt_wtm import *

# In[2]:


image = 'terraux.tif'  
#image = 'Lena.bmp' 
watermark = 'watermark.jpeg' 





# # Blind Watermarking
# In this scenario which is the combination between the wavelet transform and discreate cosine transform. At first, we do the level 2 wavelet tranform for the image and choose the aa component to apply DCT. After we have DCT image, we embedded the watermark image to this domain. Then we get the watermarked image. 
# 
# The image below is the example of the wavelet transform with level 1, 2 and 3 decomposition. The result from pywt.wavedec2 is the array of the image which separate in each section. Example, assume we have 2D coeffcient c from a level 2 transfrom, so we get 
'''
+---------------+---------------+-------------------------------+
|               |               |                               |
|     c[0]      |  c[1]['da']   |                               |
|               |               |                               |
+---------------+---------------+           c[2]['da']          |
|               |               |                               |
| c[1]['ad']    |  c[1]['dd']   |                               |
|               |               |                               |
+---------------+---------------+ ------------------------------+
|                               |                               |
|                               |                               |
|                               |                               |
|          c[2]['ad']           |           c[2]['dd']          |
|                               |                               |
|                               |                               |
|                               |                               |
+-------------------------------+-------------------------------+
'''
# In[11]:


model = 'haar'
level = 2
wm_size = 64
ori_img, image_array, image_array_rgb = image_resize(image, 2048)

#mul_wavelet_dec(image_array)
#plt.imshow(image_array)


# In[12]:


coeffs_image = process_coefficients(image_array_rgb[:,:,2], model, level=level)
#coeffs_image_l2 = process_coefficients(coeffs_image[1][0], model, level=level)

# Now we apply the DCT with the coeffs_image[0] which is the aa component in the above image. 

# In[14]:

before_wm = apply_dct(coeffs_image[0])
#before_wm = apply_dct(coeffs_image_l2[0])
#fig, axes = plt.subplots(1, 1, figsize=[10, 8])
#axes.imshow(before_wm[0:0+128, 0:0+128])
#axes.set_title("Original DCT from 2nd Coeff Wavelet Transform")
#axes.set_axis_off()



# In[15]:


ori_wm, watermark_array, watermark_rgb = image_resize(watermark, wm_size)
watermark_array = watermark_array/64
after_wm = embed_watermark(watermark_array, before_wm)

#fig, axes = plt.subplots(1, 1, figsize=[10, 8])
#axes.imshow(after_wm[0:0+128, 0:0+128])
#axes.set_title("Wavelet and DCT with Watermark")
#axes.set_axis_off()




# In[ ]:


coeffs_image_l2_temp = coeffs_image#_l2
coeffs_image_l2_temp[0][:][:] = inverse_dct(after_wm)
#temp = pywt.waverec2(coeffs_image_l2_temp, model)
#coeffs_image_l1_temp = coeffs_image
#coeffs_image_l1_temp[1][0][:][:] = temp

# In[ ]:


image_array_H=pywt.waverec2(coeffs_image_l2_temp, model)


# In[ ]:


#rgb formulate 
image_wm_array_rgb = np.zeros((2048,2048,3))
image_wm_array_rgb[:,:,2] = (image_array_H)
image_wm_array_rgb[:,:,0] = image_array_rgb[:,:,0]
image_wm_array_rgb[:,:,1] = image_array_rgb[:,:,1]


# In[ ]:


img_with_wm = print_image_from_array(image_wm_array_rgb, 'image_with_watermark')
fig, axes = plt.subplots(1, 2, figsize=[18, 10])
axes[0].imshow(ori_img)
axes[0].set_title("Image with Watermark")
axes[0].set_axis_off()

axes[1].imshow(img_with_wm)
axes[1].set_title("Original Image")
axes[1].set_axis_off()


# In[ ]:


recover_watermark(image_array = image_wm_array_rgb[:,:,2], model=model, level = level,wm_size=wm_size, name = '_no_test')
extracted_wm = Image.open('./result/recovered_watermark_no_test.bmp').resize((128, 128), 1)
extracted_wm = extracted_wm.convert('RGB')
fig, axes = plt.subplots(1, 2, figsize=[10, 5])
axes[0].imshow(extracted_wm)
axes[0].set_title("Extracted Watermark")
axes[0].set_axis_off()

axes[1].imshow(ori_wm)
axes[1].set_title("Original Watermark")
axes[1].set_axis_off()

# %% PNSR
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

#%%
    
d = psnr(image_array_rgb,image_wm_array_rgb)
print(d)

#%%
#Attack image 
