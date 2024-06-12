import cv2
import numpy as no
import os

def assure_path_exists(path):
    dir - os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

kamera = cv2.VideoCapture(1)
deteksi1_wajah =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
deteksi_mata = cv2.CascadeClassifier('harcascade_eye.xml')
face_id = 1
eye_id = 1
count = 0
assure_path_exists("dataset/")

while True:
    cond, frame = kamera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = deteksi1_wajah.detectMultiScale(gray, 1.3, 5)
    for (x,t,w,h) in wajah:
        cv2.rectangle(frame, (c,y), (x+w, y+h), (0,255,0), 5)
        count +=1

        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])
        roi_warna = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        mata = deteksi_mata.detectMultiScale(roi_gray)
        for (mx,my,mw,mh) in mata:
            cv2.reactangle(roi_warna, (mx,my) (mx+mw, my+mh), (255,255,0), 2)
        count += 1
        cv2.imwrite("dataset/User." + str(eye_id) + '.' + str(count) + "jpg", gray[y:y+h,x:x])
    cv2.imshow('Face and Eye Detection', frame)

    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        break 
    elif count>100:
        break

kamera.release()
cv2.destroyAllWindows()