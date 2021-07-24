from fer import FER
import cv2
import random
import math




import matplotlib.pyplot as plt 


img = plt.imread("./images/testing/blurry images/im2.jpg")
detector = FER(mtcnn=True)
# print(detector.detect_emotions(img))

emotion, score = detector.top_emotion(img)
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
    
print(math.ceil(score*100)/100) 
