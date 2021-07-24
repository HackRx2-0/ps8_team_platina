import os
import cv2
from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np

im = Image.open('./rotatednew.jpg')


for key,value in im._getexif().items():
   if TAGS.get(key) == 'Orientation':
      orientation = value

print(orientation)     
if orientation == 1:
   img=im
if orientation == 3:
   img = im.rotate(0)
if orientation == 6:
   img = im.rotate(360)
if orientation == 8:
   img = im.rotate(0)
  
img.save('oriented.jpg')