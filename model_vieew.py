import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
 from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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

# Define and train the deep learning model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Deep Learning Test Accuracy: {accuracy * 100:.2f}%")
model.save('gender.h5')

# Preprocess data for machine learning models
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
y_train_flat = np.argmax(y_train, axis=1)
y_test_flat = np.argmax(y_test, axis=1)

# Train SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_flat, y_train_flat)
svm_accuracy = accuracy_score(y_test_flat, svm_model.predict(X_test_flat))
print(f"SVM Test Accuracy: {svm_accuracy * 100:.2f}%")

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=84)
rf_model.fit(X_train_flat, y_train_flat)
rf_accuracy = accuracy_score(y_test_flat, rf_model.predict(X_test_flat))
print(f"Random Forest Test Accuracy: {rf_accuracy * 100:.2f}%")

# Plot the accuracies
model_names = ['Deep Learning', 'SVM', 'Random Forest']
accuracies = [accuracy * 100, svm_accuracy * 100, rf_accuracy * 100]

plt.figure(figsize=(10, 5))
plt.plot(model_names, accuracies, marker='o', linestyle='-', color='b')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracies Comparison')
plt.ylim(0, 100)
plt.yticks(np.arange(0, 101, 5))  # Set y-axis ticks every 5 points
plt.grid(True)
plt.show()
