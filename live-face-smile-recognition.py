import cv2

face_detections = cv2.CasscadeClassifier('recognition.xml')
face_smile_detections = cv2.CasscadeClassifier('smile.xml')

# Capture Video From Webcam
camera = cv2.VideoCapture(0)

while True:

    # Read the current frame
    successful_frame_read, frame = camera.read()

    # Must Convert to grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # Detect Faces
    face_infos = face_detections.detectMultiScale(gray_img)

    # Draw Green Rectangle
    for (x, y, w, h) in face_infos:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display The image and the recognition system in a windows (open-cv)
    cv2.imshow('Flora Smile Recognition', frame)

    # Key Listener
    key = cv2.waitKey(1)

    if key == 81 or key==113:
        break

# Release the camera object
camera.release()

print('code complete')