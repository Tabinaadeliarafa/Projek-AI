import cv2
import numpy as np
from mtcnn import MTCNN
from keras.models import load_model

# Load pre-trained model for age prediction
age_model = load_model('age_model.h5')  # Pastikan path ini benar

# Initialize MTCNN for face detection
detector = MTCNN()

# Age categories typically used for age prediction models
age_classes = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    faces = detector.detect_faces(frame)
    for face in faces:
        x, y, width, height = face['box']
        face_img = frame[y:y+height, x:x+width]
        face_img = cv2.resize(face_img, (64, 64))
        face_img = face_img.astype('float32') / 255
        face_img = np.expand_dims(face_img, axis=0)

        # Predict age
        age_pred = age_model.predict(face_img)
        age_index = np.argmax(age_pred)
        age_label = age_classes[age_index]

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame, age_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the output
    cv2.imshow('Age Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()