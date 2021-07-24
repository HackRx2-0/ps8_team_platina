# def jpeg_res(filename):
#    """"This function prints the resolution of the jpeg image file passed into it"""

#    # open image for reading in binary mode
#    with open(filename,'rb') as img_file:

#        # height of image (in 2 bytes) is at 164th position
#        img_file.seek(163)

#        # read the 2 bytes
#        a = img_file.read(2)

#        # calculate height
#        height = (a[0] << 8) + a[1]

#        # next 2 bytes is width
#        a = img_file.read(2)

#        # calculate width
#        width = (a[0] << 8) + a[1]

#    print("The resolution of the image is",width,"x",height)

# jpeg_res("./images/testing/edited_him.jpg")


import cv2


from PIL import Image, ImageOps
from PIL.ExifTags import TAGS

from imutils import paths
import argparse
import tensorflow.keras
import numpy as np

im = cv2.imread('./images/testing/blurry images/img8.jpeg')

# print(type(im))
# <class 'numpy.ndarray'>

# print(im.shape)
# print(type(im.shape))
# (225, 400, 3)
# <class 'tuple'>

# finding the height and width of the image 

h, w, c = im.shape
print('width:  ', w)
print('height: ', h)
print('channel:', c)

aspectR = h/w 

if aspectR >= 0.5 and aspectR <= 2 and h>100 and w>100:
    print("accepted ")
else:
    print("not accepted because you photo is low resolution")
    
    
    
np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('./converted_keras/keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
# image = Image.open('test_photo.jpg')

#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(im, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)

# display the resized image
image.show()

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
print(prediction)

    

# def variance_of_laplacian(image):
# 	# compute the Laplacian of the image and then return the focus
# 	# measure, which is simply the variance of the Laplacian
# 	return cv2.Laplacian(image, cv2.CV_64F).var()




# construct the argument parse and parse the arguments

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", required=True,
# 	help="path to input directory of images")
# ap.add_argument("-t", "--threshold", type=float, default=100.0,
# 	help="focus measures that fall below this value will be considered 'blurry'")
# args = vars(ap.parse_args())


imgRes = cv2.resize(im,None,fx=1, fy=1, interpolation = cv2.INTER_CUBIC)

# gray = cv2.cvtColor(imgRes, cv2.COLOR_BGR2GRAY)
# fm = variance_of_laplacian(gray)
# text = "Not Blurry"

# if fm < args["threshold"]:
#     text = "Blurry"

# cv2.putText(imgRes, "{}: {:.2f}".format(text, fm), (10, 30),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

# cv2.imshow("Image", imgRes)
# key = cv2.waitKey(0)

# for imagePath in paths.list_images(args["images"]):
# 	# load the image, convert it to grayscale, and compute the
# 	# focus measure of the image using the Variance of Laplacian
# 	# method
# 	image = cv2.imread(imagePath)
# 	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	fm = variance_of_laplacian(gray)
# 	text = "Not Blurry"
# 	# if the focus measure is less than the supplied threshold,
# 	# then the image should be considered "blurry"
# 	if fm < args["threshold"]:
# 		text = "Blurry"
# 	# show the image
# 	cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
# 	cv2.imshow("Image", image)
# 	key = cv2.waitKey(0)



#check the resolution of image and adjust accordingly

#condition for checking if else condition for the size of the image because we do not want to resize eveuthing 


