from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import face_recognition
import keras
from keras.models import load_model
import cv2



classifier = load_model('./model_v6_23.hdf5')

#Emotion Detection

print("Emotion Detection")
video_capture = cv2.VideoCapture(0)
emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
# model_v6_23.hdf5 ## No Lag but Not Accurate
# final_xception.h5 ## So much lag but Accurate (not currently printing out the right results tho)
model = load_model("./final_xception.h5")

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_image = cv2.resize(frame, (48,48), fx=0.25, fy=0.25)
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []




        print("Trained:")
        print (face_image.shape) #Trained prediction

        predicted_class = np.argmax(model.predict(face_image))

        label_map = dict((v,k) for k,v in emotion_dict.items())
        predicted_label = label_map[predicted_class]

        print(predicted_label)
        emotion = predicted_label


    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, emotion, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()



face_image  = cv2.imread("./test_images/index2.jpeg")
plt.imshow(face_image)
print("Untrained:")
print (face_image.shape) #Untrained prediction



#plt.show()
