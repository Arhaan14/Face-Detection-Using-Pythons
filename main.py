# import cv2
# import numpy as np

# # Load the face cascade classifier
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # Open a connection to the default camera
# cap = cv2.VideoCapture(0)

# while True:
#     # Read a frame from the camera
#     ret, frame = cap.read()

#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     # Draw a rectangle around each detected face
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         # cv2.putText(img, f"Person {i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


#     # Display the result
#     cv2.imshow("Live Object Detection", frame)

#     # Wait for a key press
#     key = cv2.waitKey(1) & 0xFF

#     # If the 'q' key is pressed, exit the loop
#     if key == ord('q'):
#         break

# # Release the camera and close the window
# cap.release()
# cv2.destroyAllWindows()





























































# import cv2
# import numpy as np

# # Load the face cascade classifier
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # Initialize the video capture object for the live camera feed
# cap = cv2.VideoCapture(0)

# # Initialize the person count variable
# person_count = 0

# while True:
#     # Read the current frame from the live camera feed
#     ret, frame = cap.read()

#     # Convert the frame to grayscale for faster processing
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the grayscale image using the Haar Cascade classifier
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#     # Loop through each detected face and draw a bounding box around it
#     for (x, y, w, h) in faces:
#         # Increment the person count for each detected face
#         person_count += 1

#         # Draw a bounding box around the detected face
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#         # Add the person count label to the bounding box
#         cv2.putText(frame, f'Person {person_count}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # Display the resulting frame
#     cv2.imshow('Face detection', frame)

#     # Exit the loop if the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture object and close all windows
# cap.release()
# cv2.destroyAllWindows()


























































import cv2
import numpy as np

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize the video capture object for the live camera feed
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame from the live camera feed
    ret, frame = cap.read()

    # Convert the frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image using the Haar Cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Initialize the person count variable
    person_count = 0

    # Loop through each detected face and draw a bounding box around it
    for (x, y, w, h) in faces:
        # Increment the person count for each detected face
        person_count += 1

        # Draw a bounding box around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Add the person count label to the bounding box
        cv2.putText(frame, f'Person {person_count}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
























































# # Load pre-trained classifier
# classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# # Read image
# img = cv2.imread("Arhaan.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Detect objects
# faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# # Draw rectangles around objects
# for (x, y, w, h) in faces:
#      cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# # Display image
# cv2.imshow("Objects", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
