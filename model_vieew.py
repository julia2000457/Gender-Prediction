import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def preprocess_image(image_path, size=(64, 64)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, size)  # Resize image
    image = image / 255.0  # Normalize pixel values
    return image


def load_dataset(dataset_path):
    images = []
    labels = []
    for gender in ["man", "woman"]:
        gender_path = os.path.join(dataset_path, gender)
        label = 0 if gender == "man" else 1
        for image_name in os.listdir(gender_path):
            image_path = os.path.join(gender_path, image_name)
            image = preprocess_image(image_path)
            images.append(image)
            labels.append(label)

    images = np.array(images).reshape(-1, 64, 64, 1)  # Reshape for model
    labels = to_categorical(np.array(labels), num_classes=2)  # One-hot encode labels
    return train_test_split(images, labels, test_size=0.2, random_state=84)


# Load dataset
dataset_path = 'dataset1/train'
X_train, X_test, y_train, y_test = load_dataset(dataset_path)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 output units for 2 classes
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the trained model

model.save('gender.h5')