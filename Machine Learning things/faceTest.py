from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import face_recognition
import keras
from keras.models import load_model
import cv2



classifier = load_model('./model_v6_23.hdf5')

#Face Recognition

print("Face Recognition")

image1 = Image.open("./test_images/index2.jpeg")
image_array1 = np.array(image1)
plt.imshow(image_array1)

image2 = Image.open("./test_images/index1.jpg")
image_array2 = np.array(image2)
plt.imshow(image_array2)

image3 = Image.open("./test_images/rajeev.jpg")
image_array3 = np.array(image3)
plt.imshow(image_array3)

image1 = face_recognition.load_image_file("./test_images/index1.jpg")
image2 = face_recognition.load_image_file("./test_images/rajeev.jpg")

encoding_1 = face_recognition.face_encodings(image1)[0]

encoding_2 = face_recognition.face_encodings(image1)[0]

results = face_recognition.compare_faces([encoding_1], encoding_2,tolerance=0.50)

print (results)

#Emotion Detection

print("Emotion Detection")
video_capture = cv2.VideoCapture(0)
emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

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



        model = load_model("./model_v6_23.hdf5")
        print("Trained:")
        print (face_image.shape) #Trained prediction

        predicted_class = np.argmax(model.predict(face_image))

        label_map = dict((v,k) for k,v in emotion_dict.items())
        predicted_label = label_map[predicted_class]

        print(predicted_label)
        emotion = predicted_label

        #for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            #matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            #emotion = predicted_label

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            #face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            #best_match_index = np.argmin(face_distances)
            #if matches[best_match_index]:
            #    name = known_face_names[best_match_index]

            #face_names.append(name)

    process_this_frame = not process_this_frame

    #model = load_model("./emotion_detector_models/model.hdf5")
    #predicted_class = np.argmax(model.predict(frame))
    #print(pred)


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



plt.show()
