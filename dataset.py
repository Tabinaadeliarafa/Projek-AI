import cv2
import numpy as np
import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

cam = cv2.VideoCapture(0)
deteksi1_wajah = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
deteksi_mata = cv2.CascadeClassifier('haarcascade_eye.xml')
face_id = 1
count = 0
assure_path_exists("dataset/")

while True:
    cond, frame = cam.read()
    if not cond:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = deteksi1_wajah.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in wajah:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)
        count += 1

        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])
        roi_warna = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        mata = deteksi_mata.detectMultiScale(roi_gray)
        for (mx, my, mw, mh) in mata:
            cv2.rectangle(roi_warna, (mx, my), (mx+mw, my+mh), (255, 255, 0), 2)
                
    cv2.imshow('Face and Eye Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('Face and Eye Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

cam.release()
cv2.destroyAllWindows()
print("Dataset collected successfully")