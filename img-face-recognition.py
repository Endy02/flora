import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv ( haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('recognition.xml') 

# Image to detect faces in
img = cv2.imread('tkt.jpg')


# Must Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 


# Detect Faces
face_infos = trained_face_data.detectMultiScale(gray_img)

# Display face coordinates
print(face_infos)

# Draw Green Rectangle
for (x, y, w, h) in face_infos:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(128,256), randrange(128,256), randrange(128,256)), 2)


# Display The image and the recognition system in a windows (open-cv)
cv2.imshow('Flora Image Recognition', img)

# Key Listener
cv2.waitKey()

print('End of System')

