# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 20:36:30 2024

@author: GS
"""
import numpy as np
from PIL import Image
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
lena_image ="lenna.png"
image = Image.open(lena_image)
lenna_array = np.array(image)
lenna_blue = lenna_array.copy()
lenna_blue[:,:,0]=0
lenna_blue[:,:,1]=0
blue_image = Image.fromarray(lenna_blue)
concatenated_image = get_concat_h(image,blue_image)
concatenated_image.show()
