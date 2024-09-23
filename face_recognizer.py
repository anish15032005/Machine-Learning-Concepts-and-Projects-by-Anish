# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model

# Load the Haar cascade classifier for face detection
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Set the video capture properties
cap.set(3, 640)  # Set the width to 640 pixels
cap.set(4, 480)  # Set the height to 480 pixels

# Define the font for text rendering
font = cv2.FONT_HERSHEY_COMPLEX

# Load the pre-trained Keras model
model = load_model('keras_model.h5')

# Define a function to get the class name based on the class index
def get_className(classNo):
    if classNo == 0:
        return "Chando"
    elif classNo == 1:
        return "Tony Stark"

# Main loop
while True:
    # Read a frame from the video capture object
    success, imgOrignal = cap.read()

    # Detect faces in the frame using the Haar cascade classifier
    faces = facedetect.detectMultiScale(imgOrignal, 1.3, 5)

    # Loop through each detected face
    for x, y, w, h in faces:
        # Crop the face region from the original image
        crop_img = imgOrignal[y:y+h, x:x+h]

        # Resize the cropped face image to 224x224 pixels
        img = cv2.resize(crop_img, (224, 224))

        # Reshape the resized image to a 4D tensor (batch size, height, width, channels)
        img = img.reshape(1, 224, 224, 3)

        # Make a prediction on the face image using the pre-trained model
        prediction = model.predict(img)

        # Get the class index of the predicted class
        classIndex = model.predict_classes(img)

        # Get the probability value of the predicted class
        probabilityValue = np.amax(prediction)

        # Draw a rectangle around the face region
        if classIndex == 0:
            cv2.rectangle(imgOrignal, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y-40), (x+w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        elif classIndex == 1:
            cv2.rectangle(imgOrignal, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y-40), (x+w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

        # Display the probability value
        cv2.putText(imgOrignal, str(round(probabilityValue*100, 2)) + "%", (180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the output image
    cv2.imshow("Result", imgOrignal)

    # Check for the 'q' key press to exit the loop
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()