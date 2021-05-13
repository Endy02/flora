# OpenCV
import cv2

# Face detector
face_detector = cv2.CascadeClassifier('recognition.xml')

# Smile detector
face_smile_detector = cv2.CascadeClassifier('smile.xml')

# Eyes detector
face_eyes_detector = cv2.CascadeClassifier('eyes.xml')

# Capture Video From Webcam
camera = cv2.VideoCapture(0)

# Loop and wait till the key is press
while True:

    # Read the current frame
    successful_frame_read, frame = camera.read()

    # if there's an error, abort
    if not successful_frame_read:
        break

    # Must Convert to grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # Detect Faces
    face_infos = face_detector.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)

    # Draw Green Rectangle
    for (x, y, w, h) in face_infos:
        # Draw Rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #get the face frame and Dimensional array slicing
        face_ = frame[y:y+h, x:x+w]

        # Must Convert to grayscale
        gray_face = cv2.cvtColor(face_, cv2.COLOR_BGR2GRAY)

        # Detect Smile
        smile_infos = face_smile_detector.detectMultiScale(gray_face, scaleFactor=1.5, minNeighbors=35)
        
        # Detect Eyes
        eye_infos = face_eyes_detector.detectMultiScale(gray_face)

        # if smile detected
        if len(smile_infos) > 0:
            # Face label
            cv2.putText(frame, 'Smiling', (x, y-25), fontScale=2, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))
        # Find all smiles in the face
        for (x_, y_, w_, h_) in smile_infos:

            # Draw Rectangle
            cv2.rectangle(face_, (x_, y_), (x_+w_, y_+h_), (50, 50, 200), 2)

    # Display The image and the recognition system in a windows (open-cv)
    cv2.imshow('Flora Smile Recognition', frame)

    # Key Listener
    key = cv2.waitKey(1)

    # "Q/q" key press to quit the script
    if key == 81 or key==113:
        break

# Release the camera object
camera.release()
cv2.destroyAllWindows()

# End Of Script
print('End of System')