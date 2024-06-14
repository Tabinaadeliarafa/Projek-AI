import cv2
import numpy as np
import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

recognizer = cv2.face.LBPHFaceRecognizer_create()
assure_path_exists("Trainer/")
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)

while True:
    ret, im = cam.read()
    if not ret:
        break
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in faces:
        cv2.rectangle(im, (x-20, y-20), (x+w+20, y+h+20), (252, 192,203), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        if id == 1:
            id_text = "siti(18thn) {:. 2f}%".format(round(100 - confidence, 2))
        else:
            id_text = "unknown {:. 2f}%".format(round(100 - confidence, 2))
        cv2.rectangle(im, (x-22, y-90),(x+w+22, y-22), (252,192,203), -1)
        cv2.putText(im, str(Id), (x-0,y-40), font, 1, (0,0,0), 3)

    cv2.imshow('Deteksi mata dan wajah', im)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('Deteksi mata dan wajah', cv2.WND_PROP_VISIBLE) < 1:
        break

cam.release()
cv2.destroyAllWindows()