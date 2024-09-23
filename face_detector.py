import cv2
import pygame

# Initialize Pygame mixer
pygame.mixer.init()

# Load the sound
sound = pygame.mixer.Sound("./notification_popup.wav")

# Ensure the correct path to the Haar cascade file
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cap = cv2.CascadeClassifier(face_cascade_path)

# This variable is for opening camera and capturing the video
video_cap = cv2.VideoCapture(0)  # This is for enabling camera at runtime

if not video_cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

face_detected = False

while True:
    ret, video_data = video_cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) > 0:
        if not face_detected:
            print("Face detected!")
            sound.play()
            face_detected = True
    else:
        face_detected = False

    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("video_live", video_data)

    if cv2.waitKey(10) == ord("a"):
        break

video_cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
