from PIL import Image
import cv2
import numpy as np


im = Image.open('./blur4.jpg')
img = np.array(im, dtype=np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 


print((np.max(cv2.convertScaleAbs(cv2.Laplacian(gray,3)))))
if(np.max(cv2.convertScaleAbs(cv2.Laplacian(gray,3))))<50:
    print("blurry")
else:
    print("good")