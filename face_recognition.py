import cv2
import numpy as np

# Load the trained LBPH face recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('path_to_your_model/lbph_face_recognizer.yml')

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to recognize faces in a frame
def recognize_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        label, confidence = recognizer.predict(face_roi)

        if label == 1 and confidence < 100:  # Assuming "1" is your label for Shaheer
            name = "Shaheer"
        else:
            name = "Unknown"

        # Display result
        color = (0, 255, 0) if name == "Shaheer" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame

# Start video capture from webcam
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Recognize face
    frame = recognize_face(frame)

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
