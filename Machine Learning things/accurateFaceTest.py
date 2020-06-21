from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import face_recognition
import keras
from keras.models import load_model
import cv2
from imutils import face_utils



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

    detectedFace = detectFace(frame)



    prediction = model.predict(face_image)
    prediction_result = np.argmax(prediction)



    def extract_face_features(faces, offset_coefficients=(0.075, 0.05)):
        gray = faces[0]
        detected_face = faces[1]

        new_face = []

        for det in detected_face :
            #Region dans laquelle la face est détectée
            x, y, w, h = det
            #X et y correspondent à la conversion en gris par gray, et w, h correspondent à la hauteur/largeur

            #Offset coefficient, np.floor takes the lowest integer (delete border of the image)
            horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
            vertical_offset = np.int(np.floor(offset_coefficients[1] * h))

            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #gray transforme l'image
            extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]

            #Zoom sur la face extraite
            new_extracted_face = zoom(extracted_face, (shape_x / extracted_face.shape[0],shape_y / extracted_face.shape[1]))
            #cast type float
            new_extracted_face = new_extracted_face.astype(np.float32)
            #scale
            new_extracted_face /= float(new_extracted_face.max())
            #print(new_extracted_face)

            new_face.append(new_extracted_face)

        return new_face


    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

    (eblStart, eblEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (ebrStart, ebrEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

    # 1. Add prediction probabilities
    cv2.putText(frame, "Angry : " + str(round(prediction[0][0],3)),(40,140 + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
    cv2.putText(frame, "Disgust : " + str(round(prediction[0][1],3)),(40,160 + 180*2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
    cv2.putText(frame, "Fear : " + str(round(prediction[0][2],3)),(40,180 + 180*3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
    cv2.putText(frame, "Happy : " + str(round(prediction[0][3],3)),(40,200 + 180*4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
    cv2.putText(frame, "Sad : " + str(round(prediction[0][4],3)),(40,220 + 180*5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
    cv2.putText(frame, "Surprise : " + str(round(prediction[0][5],3)),(40,240 + 180*6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
    cv2.putText(frame, "Neutral : " + str(round(prediction[0][6],3)),(40,260 + 180*7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)

    # 2. Annotate main image with a label
    if prediction_result == 0 :
        cv2.putText(frame, "Angry",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif prediction_result == 1 :
        cv2.putText(frame, "Disgust",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif prediction_result == 2 :
        cv2.putText(frame, "Fear",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif prediction_result == 3 :
        cv2.putText(frame, "Happy",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif prediction_result == 4 :
        cv2.putText(frame, "Sad",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif prediction_result == 5 :
        cv2.putText(frame, "Surprise",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else :
        cv2.putText(frame, "Neutral",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


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



def detectFace(frame):

    #Cascade classifier pre-trained model
    cascPath = 'face_landmarks.dat'
    faceCascade = cv2.CascadeClassifier(cascPath)

    #BGR -> Gray conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Cascade MultiScale classifier
    detected_faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6,
                                                  minSize=(shape_x, shape_y),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)
    coord = []

    for x, y, w, h in detected_faces :
        if w > 100 :
            sub_img=frame[y:y+h,x:x+w]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255,255),1)
            coord.append([x,y,w,h])

    return gray, detected_faces, coord



#plt.show()
