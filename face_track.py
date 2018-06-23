import cv2 # Open Source Computer Vision Library
import sys # System-specific parameters and functions

# haarcascade Object Detection using Haar feature-based cascade classifiers. Sorts face vs non face images and determines
# which are which in the camera stream.

faceCascade = cv2.CascadeClassifier("haarcascade.xml")
video_capture = cv2.VideoCapture(0)

while True:

    # Capture video frame-by-frame
    returnVal, frame = video_capture.read()

    # Convert frames to grayscale to speed up image processing
    grayScaleConvert = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect facial feature using Haar Cascade
    face = faceCascade.detectMultiScale(
        grayScaleConvert,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(35, 35)
    )

    # Draw the facial rectangle box
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 200), 2)

    # Display result
    cv2.imshow('Video', frame)

    # Exit camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
       sys.exit() 