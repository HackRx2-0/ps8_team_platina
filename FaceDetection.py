
#import the libraries

import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2
import numpy as np
from PIL.ExifTags import TAGS
import argparse
import json
import random
import math
import matplotlib.pyplot as plt 
from fer import FER
import imageio
import sys
import math

#functions


def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()


def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val  = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil( n_cols * (1.0 - half_percent))]

        print(low_val)
        print(high_val)

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

ResponseValue = {'AspectRatio':''}



def checkifHuman(image):
    # image = Image.open(filename)
    np.set_printoptions(suppress=True)
    HumanAvtars = tensorflow.keras.models.load_model('./ml-model/avtar-cartoon-human/keras_model.h5',compile=False)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = np.array(HumanAvtars.predict(data))
    if prediction[0][0] >= prediction[0][1]:
        return 0
    else:
        return 1
    

def occlusion(image):

    np.set_printoptions(suppress=True)
    Occlusion = tensorflow.keras.models.load_model('./ml-model/Occlusion/keras_model.h5',compile=False)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = np.array(Occlusion.predict(data))
    if prediction[0][0] > prediction[0][1]:
        return 0
    else:
        return 1
    
    
    

def watermarks(image):

    np.set_printoptions(suppress=True)
    WaterMarks = tensorflow.keras.models.load_model('./ml-model/watermark/keras_model.h5',compile=False)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    
    prediction = np.array(WaterMarks.predict(data))
    print(prediction)
    if prediction[0][0] > prediction[0][1]:
        return 1
    else:
        return 0
    
    
    
def PrintedNonPrinted(image):

    np.set_printoptions(suppress=True)
    WaterMarks = tensorflow.keras.models.load_model('./ml-model/printed-nonprinted/converted_keras (1)/keras_model.h5',compile=False)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    
    prediction = np.array(WaterMarks.predict(data))
    print(prediction)
    if prediction[0][0] > prediction[0][1]:
        return 1
    else:
        return 0
    

    

def FitMentScore(image):
    detector = FER(mtcnn=True)
    emotion, score = detector.top_emotion(image)
    print(emotion,score)

    if emotion == "happy":
        score = 85 + score*random.uniform(8, 10)
    elif emotion == "neutral":
        score = 76 + score*random.uniform(5, 7)
    elif emotion == "suprise":
        score = 63 + score*random.uniform(5, 7)
    elif emotion == "sad":
        score = 20 + score*random.uniform(11,20)
    elif emotion == "fear":
        score = 13 + score*random.uniform(5, 10)

    elif emotion == "disgust":
        score = 3 + score*random.uniform(5, 10)

    else:
        score = random.uniform(10)
    
    facialexpression = emotion
    fitmentscore = (math.ceil(score*100)/100)
    
    value = {
        "facialexpression" : facialexpression,
        "fitmentscore" : fitmentscore
    }
    
    return value
    

def img_estim(image, thrshld):
    is_light = np.mean(image) > thrshld
    return 'light' if is_light else 'dark'
    
            
        
# img = cv2.imread("./images/testing/images/overexposed/img3.jpg")
# out = simplest_cb(img, 15)
# cv2.imshow("before", img)
# cv2.imshow("after", out)           
        
        
   

# def img_estim(img, thrshld):
#     is_light = np.mean(img) > thrshld
#     return 'light' if is_light else 'dark'

# print(img_estim(f, 127))


demofinal = './images/testing/images/overexposed/img1.jpg'

def CheckImage():
    image = cv2.imread(demofinal)
    h, w, c = image.shape
    area = h*w
    print('width:  ', w)
    print('height: ', h)
    print('channel:', c)
    
    f = imageio.imread(demofinal, as_gray=True)
    checking = img_estim(f, 127)
    out = simplest_cb(image, 15)
    cv2.imshow("before", image)
    cv2.imshow("after", out)
    cv2.waitKey(0)
    ResponseValue.update(LightQuality = checking)
    aspectR = h/w 

    if aspectR >= 0.5 and aspectR <= 2 and h>100 and w>100:
         ResponseValue.update(AspectRatio =aspectR)
        
         im = Image.open(demofinal)
         print(checkifHuman(im))
         if checkifHuman(im):
            ResponseValue.update(HumanTest ='Pass')
            if occlusion(im):
                # im.show()
                ResponseValue.update(Status ='Rejected')
                ResponseValue.update(Reason ='Occlusion Detected')
                return ResponseValue
                
            else:
                ResponseValue.update(OcclusionTest ='Pass')
                im.show()
                if PrintedNonPrinted(im):
                    ResponseValue.update(Status ='Rejected')
                    ResponseValue.update(Reason ='Print Detected')
                    return ResponseValue
                else:
                    # faceAreaandCrop(im,area)
                    ResponseValue.update(PrintTest ='Pass')
                    fitment = FitMentScore(image)
                    ResponseValue.update(fitment)
                    
                    return ResponseValue
                    
                    
                
                
                    
                
            
             
         else:
             ResponseValue.update(Status ='Rejected')
             ResponseValue.update(Reason ='not a human')
             return ResponseValue
        
        
       
    else:
        ResponseValue.update(AspectRatio =aspectR)
        ResponseValue.update(Status ='not accpeted')
        
        return ResponseValue
    
    


response = CheckImage()

print(response)