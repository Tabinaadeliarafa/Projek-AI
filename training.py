import cv2, os
import numpy as np
from PIL import Image 
import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.exists(dir):
        os.makedirs(dir)

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector - cv2.cascadeClassifier("haarcascade_frontalface_default.xml");
def getImageAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_ing = Image.open(ImagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids

faces,ids = getImageAndLabels('dataset')
recognizer.train(faces, np.array(ids))
assure_path_exists('traoner/')
recognizer.save('traineer/')
recognizer.save('trainer/trainer.yml')
                                                