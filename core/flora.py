import cv2
from random import randrange
from tkinter import *

class Flora:
    trained_face = cv2.CascadeClassifier('./data/recognition.xml')
    trained_smile = cv2.CascadeClassifier('./data/smile.xml')
    trained_eyes = cv2.CascadeClassifier('./data/eyes.xml') 

    def __init__(self):
        self.flora_interface()

    def flora_interface(self):
        # Construction de la fenêtre principale «root»
        root = Tk()
        root.title('Flora Home')
        
        # Configuration du gestionnaire de grille
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        # Construction d'un simple bouton
        qb = Button(root, text='Quitter', command=root.quit)

        # Placement du bouton dans «root»
        # Placement du bouton dans «root»
        qb.grid(row=0, column=0, sticky="nsew")

        # Lancement de la «boucle principale»
        root.mainloop()

    def face_recognition(self):
        """
            Function to detect faces with default camera
        """
        camera = cv2.VideoCapture(0)
        
        while True:
            # Read the current frame
            successful_frame_read, frame = camera.read()
            # if there's an error, abort
            if not successful_frame_read:
                break
            # Must Convert to grayscale
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            # Detect Faces
            face_infos = self.trained_face.detectMultiScale(gray_img)
            # Draw Green Rectangle
            for (x, y, w, h) in face_infos:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display The image and the recognition system in a windows (open-cv)
            cv2.imshow('Flora', frame)
            # Key Listener
            key = cv2.waitKey(1)

            if key == 81 or key==113:
                break
        # Release the camera object
        camera.release()
        
    def smile_detection(self):
        """
            Function to detect smille with interne camera
        """
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
            face_infos = self.trained_face.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)

            # Draw Green Rectangle
            for (x, y, w, h) in face_infos:
                # Draw Rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                #get the face frame and Dimensional array slicing
                face_ = frame[y:y+h, x:x+w]
                # Must Convert to grayscale
                gray_face = cv2.cvtColor(face_, cv2.COLOR_BGR2GRAY)
                # Detect Smile
                smile_infos = self.trained_smile.detectMultiScale(gray_face, scaleFactor=1.2, minNeighbors=55)
                # Detect Eyes
                eye_infos = self.trained_eyes.detectMultiScale(gray_face)
                # if smile detected
                if len(smile_infos) > 0:
                    # Face label
                    cv2.putText(frame, 'Smiling', (x, y-25), fontScale=2, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))
                # Find all smiles in the face
                for (x_, y_, w_, h_) in smile_infos:

                    # Draw Rectangle
                    cv2.rectangle(face_, (x_, y_), (x_+w_, y_+h_), (50, 50, 200), 2)

            # Display The image and the recognition system in a windows (open-cv)
            cv2.imshow('Flora', frame)

            # Key Listener
            key = cv2.waitKey(1)

            # "Q/q" key press to quit the script
            if key == 81 or key==113:
                break

        # Release the camera object
        camera.release()
        cv2.destroyAllWindows()

    def img_face_recognition(self):
        """
            Function to scan faces in an given image
        """
        img = cv2.imread('./assets/images/test_face_2.jpg')
        # Must Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect Faces
        face_infos = self.trained_face.detectMultiScale(gray_img)

        # Draw Green Rectangle
        for (x, y, w, h) in face_infos:
            cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(128,256), randrange(128,256), randrange(128,256)), 2)

        # Display The image and the recognition system in a windows (open-cv)
        cv2.imshow('Flora', img)

        # Key Listener
        cv2.waitKey()