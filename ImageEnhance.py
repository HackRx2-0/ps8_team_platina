from PIL import Image, ImageEnhance
import numpy as np
import cv2

import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imshow, imread


import imageio

from skimage.util import img_as_ubyte

img = cv2.imread("./images/testing/images/overexposed/img2.jpg")



def gamma_trans(img, gamma):
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)

image_gamma_correct=gamma_trans(img,0.1)

import math

# read image
# img = cv2.imread('lioncuddle1.jpg')

# METHOD 1: RGB

# convert img to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# compute gamma = log(mid*255)/log(mean)
mid = 0.3
mean = np.mean(gray)
gamma = math.log(mid*255)/math.log(mean)
print(gamma)

# do gamma correction
img_gamma1 = np.power(img, gamma).clip(0,255).astype(np.uint8)



# METHOD 2: HSV (or other color spaces)

# convert img to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue, sat, val = cv2.split(hsv)

# compute gamma = log(mid*255)/log(mean)
mid = 0.4
mean = np.mean(val)
gamma = math.log(mid*255)/math.log(mean)
print(gamma)

# do gamma correction on value channel
val_gamma = np.power(val, gamma).clip(0,255).astype(np.uint8)

# combine new value channel with original hue and sat channels
hsv_gamma = cv2.merge([hue, sat, val_gamma])
img_gamma2 = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

# show results
cv2.imshow('input', img)
cv2.imshow('result1', img_gamma1)
cv2.imshow('result2', img_gamma2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save results
# cv2.imwrite('lioncuddle1_gamma1.jpg', img_gamma1)
# cv2.imwrite('lioncuddle1_gamma2.jpg', img_gamma2)







# save results
# cv2.imwrite('lioncuddle1_gamma1.jpg', img_gamma1)
# cv2.imwrite('lioncuddle1_gamma2.jpg', img_gamma2)














# def adjust_gamma(image, gamma=1.0):
# 	# build a lookup table mapping the pixel values [0, 255] to
# 	# their adjusted gamma values
# 	invGamma = 1.0 / gamma
# 	table = np.array([((i / 255.0) ** invGamma) * 255
# 		for i in np.arange(0, 256)]).astype("uint8")
# 	# apply gamma correction using the lookup table
    
# 	return cv2.LUT(image, table)

# adjust_gamma(img)
# cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
# cv2.imshow("img",img)
# cv2.waitKey()



# def grayHist(img):
    
#     h, w = img.shape[:2]
#     pixelSequence = img.reshape([h * w, ])
#     numberBins = 256
#     histogram, bins, patch = plt.hist(pixelSequence, numberBins,
#                                       facecolor='black', histtype='bar')
#     plt.xlabel("gray label")
#     plt.ylabel("number of pixels")
#     plt.axis([0, 255, 0, np.max(histogram)])
#     plt.show()
 
# img = cv2.imread("./images/dark1.jpeg", 0)
# out = 2.0 * img
#  # , the value greater than 255 is truncated to 255
# out[out > 255] = 255
#  # 
# out = np.around(out)
# out = out.astype(np.uint8)
#  # respectively plot the histogram before and after processing
# # grayHist(img)
# # grayHist(out)
# cv2.imshow("img", img)
# cv2.imshow("out", out)
# cv2.waitKey()


 
# img = cv2.imread("./images/dark1.jpeg", 0)
#  # Calculate the minimum gray level and maximum gray level that appear in the original image
#  # Using function calculation
# Imin, Imax = cv2.minMaxLoc(img)[:2]
#  # Calculate using numpy
# # Imax = np.max(img)
# # Imin = np.min(img)
# Omin, Omax = 0, 255
#  # Calculate the values ​​of a and b
# a = float(Omax - Omin) / (Imax - Imin)
# b = Omin - a * Imin
# out = a * img + b
# out = out.astype(np.uint8)
# out = cv2.cvtColor(out,cv2.COLOR_GRAY2RGB)
# cv2.imshow("img", img)
# cv2.imshow("out", out)
# cv2.waitKey()


#this is it


# f = imageio.imread("./images/testing/images/overexposed/img1.jpg", as_gray=True)

# def img_estim(img, thrshld):
#     is_light = np.mean(img) > thrshld
#     return 'light' if is_light else 'dark'

# print(img_estim(f, 127))








# image_overcast = imread('./images/testing/images/overexposed/img1.jpg')


# rgb_list = ['Reds','Greens','Blues']
# fig, ax = plt.subplots(1, 3, figsize=(15,5), sharey = True)
# for i in range(3):
#    ax[i].imshow(image_overcast[:,:,i], cmap = rgb_list[i])
#    ax[i].set_title(rgb_list[i], fontsize = 15)


 
 
   
   
   





























# im = Image.open("./images/testing/blurry images/img10.jpeg")

# enhancer = ImageEnhance.Sharpness(im)


# factor = 3
# im_s_1 = enhancer.enhance(factor)
# im.show()
# im_s_1.show()


# # Compute the exposure times in seconds
# exposures = np.float32([1. / t for t in [1000, 500, 100, 50, 10]])

# # Compute the response curve
# calibration = cv2.createCalibrateDebevec()
# response = calibration.process(im, exposures)

# def adjust_brightness(input_image, output_image, factor):
#     image = Image.open(input_image)
#     enhancer_object = ImageEnhance.Brightness(image)
#     out = enhancer_object.enhance(factor)
#     out.save(output_image)
    