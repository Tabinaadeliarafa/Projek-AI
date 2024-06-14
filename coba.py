import cv2
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import Loss
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class CustomMeanSquaredError(Loss):
    def __init__(self, name='custom_mse'):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

try:
    # If using the custom loss function, load the model with custom_objects argument
    age_model = load_model('age_model.h5', custom_objects={'CustomMeanSquaredError': CustomMeanSquaredError()})
except OSError:
    print("Error: Could not find 'age_model.h5'. Please check the file path.")
    exit()

# Initialize MTCNN for face detection
detector = MTCNN()

# Age categories typically used for age prediction models
age_classes = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Start video capture from the webcam
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Detect faces in the frame
    faces = detector.detect_faces(frame)
    for face in faces:
        x, y, width, height = face['box']
        face_img = frame[y:y+height, x:x+width]

        # Preprocess the face image for age prediction
        face_img = cv2.resize(face_img, (64, 64))  # Resize to 64x64 pixels
        face_img = face_img.astype('float32') / 255  # Normalize pixel values
        face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension

        # Predict age
        age_pred = age_model.predict(face_img)
        age_index = np.argmax(age_pred)
        age_label = age_classes[age_index]

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame, age_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the output
    cv2.imshow('Age Detection', frame)

    # Break the loop if 'q' key is pressed or window is closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('Age Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the capture and destroy all windows
cam.release()
cv2.destroyAllWindows()