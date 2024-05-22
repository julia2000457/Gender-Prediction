import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('gender.h5')

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Function to preprocess the image
def preprocess_image(image, size=(64, 64)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, size)  # Resize image
    image = image / 255.0  # Normalize pixel values
    image = np.reshape(image, (1, size[0], size[1], 1))  # Reshape for model
    return image


# Function to predict gender
def predict_gender(model, image):
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)
    return "male" if predicted_class == 0 else "female"



# Open a connection to the webcam
cap = cv2.VideoCapture("http://192.168.187.156:8080/video")

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        break

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y + h, x:x + w]

        # Preprocess the face for prediction
        preprocessed_face = preprocess_image(face)

        if preprocessed_face is not None:
            # Predict gender
            gender = predict_gender(model, preprocessed_face)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 100, 0), 7)
            # Put the gender label above the rectangle
            cv2.putText(frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Gender Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
