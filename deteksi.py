import cv2
import numpy as no
import os

def assure_path_exists(path):
    dir - os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

recognizer = cv2.face.LBPHFaceRegnizer_create()
assure_path_exists("Trainer/")
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEXcam = cv2.VideoCapture(1)

while True:
    ret, im =cam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    for(x,y,w,h) in fces:
        cv2.rectangle(im, (x-20,y-20), (x+w+20, y+h_20), (252, 192,203), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        if(id == 1):
            id = "siti(18thn) {:. 2f}%".format(round(100 - confidence, 2))
        cv2.rectangle(im, (x-22, y-90), )