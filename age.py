import cv2
import numpy as np
import os

# Path to model files
age_prototxt = 'models/deploy_age.prototxt'
age_model = 'models/age_net.caffemodel'
gender_prototxt = 'models/deploy_gender.prototxt'
gender_model = 'models/gender_net.caffemodel'

# Load model files
age_net = cv2.dnn.readNetFromCaffe(age_prototxt, age_model)
gender_net = cv2.dnn.readNetFromCaffe(gender_prototxt, gender_model)

# Mean values and age/gender lists
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Initialize webcam
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    # Predict gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]

    # Predict age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]

    # Display results
    label = f"{gender}, {age}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Age and Gender Recognition', frame)

    # Break loop on 'q' key pressed or window is closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('Age and Gender Recognition', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
