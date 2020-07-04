from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import face_recognition
import keras
from keras.models import load_model
import cv2
import pandas as pd
from imutils import face_utils
from scipy.ndimage import zoom
from scipy.spatial import distance
import imutils
from scipy import ndimage
import dlib
from time import process_time


cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cascade_eye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
cascade_smile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

global shape_x
global shape_y
global input_shape
global nClasses

def detection(actual_emotion):
    shape_x = 48
    shape_y = 48
    input_shape = (shape_x, shape_y, 1)
    nClasses = 7

    thresh = 0.25
    frame_check = 20
    def eye_aspect_ratio(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    def detect_face(frame):

        #Cascade classifier pre-trained model
        cascPath = 'Model/face_landmarks.dat'
        faceCascade = cv2.CascadeClassifier(cascPath)

        #BGR -> Gray conversion
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Cascade MultiScale classifier
        detected_faces = faceCascade.detectMultiScale(grayscale,scaleFactor=1.1,minNeighbors=6,
                                                      minSize=(shape_x, shape_y),
                                                      flags=cv2.CASCADE_SCALE_IMAGE)
        coord = []

        for x, y, w, h in detected_faces :
            if w > 100 :
                sub_img=frame[y:y+h,x:x+w]
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255,255),1)
                coord.append([x,y,w,h])

        return grayscale, detected_faces, coord

    def extract_face_features(faces, offset_coefficients=(0.075, 0.05)):
        grayscale = faces[0]
        detected_face = faces[1]

        new_face = []

        for det in detected_face :

            x, y, w, h = det

            horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
            vertical_offset = np.int(np.floor(offset_coefficients[1] * h))

            extracted_face = grayscale[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]

            new_extracted_face = zoom(extracted_face, (shape_x / extracted_face.shape[0],shape_y / extracted_face.shape[1]))

            new_extracted_face = new_extracted_face.astype(np.float32)

            new_extracted_face /= float(new_extracted_face.max())


            new_face.append(new_extracted_face)

        return new_face


    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

    (eblStart, eblEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (ebrStart, ebrEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

    model = load_model('Model/video.h5')
    face_detect = dlib.get_frontal_face_detector()
    predictor_landmarks  = dlib.shape_predictor("Model/face_landmarks.dat")

    vc = cv2.VideoCapture(0)
    start = process_time()
    skip_frame = True

    #Call function in loop and get it to output guess

    while True:
        _, img = vc.read()
        image = img[:]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image, 'RGB')
        #image.show()
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = face_detect(grayscale,1)

        if skip_frame:
            for (i, rect) in enumerate(rects):
                shape = predictor_landmarks(grayscale,rect)
                shape = face_utils.shape_to_np(shape)

                (x,y,w,h)= face_utils.rect_to_bb(rect) #Face coordinates
                face = grayscale[y:y+h,x:x+w]

                face = zoom(face, (shape_x / face.shape[0],shape_y / face.shape[1]))

                face = face.astype(np.float32)

                face /= float(face.max())
                face = np.reshape(face.flatten(), (1,48,48,1))

                emotion_prediction = emotionPrediction(face,model)
                if(emotion_prediction == actual_emotion):
                    end = process_time()
                    time = end - start
                    print(time)
                    frame = cv2.cvtColor(grayscale,cv2.COLOR_GRAY2RGB)
                    return time, image



        skip_frame = not skip_frame

        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vc.release()
    cv2.destroyAllWindows()


def emotionPrediction(face,model):

    emotion_dict= {'Angry': 0, 'Sad': 4, 'Neutral': 6, 'Disgust': 1, 'Surprise': 5, 'Fear': 2, 'Happy': 3}

    emotion_prediction = model.predict(face)
    deciphered_emotion = predictionDecipher(emotion_prediction)

    integer_prediction = np.argmax(emotion_prediction)

    label_map = dict((v,k) for k,v in emotion_dict.items())
    label = label_map[integer_prediction]

    print(label)
    #print()

    return label

def predictionDecipher(model_guess):
    anger_metric = str(round(model_guess[0][0],3)) #anger
    disgust_metric = str(round(model_guess[0][1],3)) #disgust
    fear_metric = str(round(model_guess[0][2],3)) #fear
    happy_metric = str(round(model_guess[0][3],3)) #happy
    sad_metric = str(round(model_guess[0][4],3)) #sad
    surprise_metric = str(round(model_guess[0][5],3)) #surprise
    neutral_metric = str(round(model_guess[0][6],3)) #neutral
    #print("Anger:"+anger_metric)
    #print("Disgust:"+disgust_metric)
    #print("Fear:"+fear_metric)
    #print("Happy:"+happy_metric)
    #print("Sad:"+sad_metric)
    #print("Surprise:"+surprise_metric)
    #print("Neutral:"+neutral_metric)

def main(actual_emotion):
    results = detection(actual_emotion) # returns true when user's emotion matches actual_emotion

    time = results[0]

    #image = Image.fromarray(results[1], 'RGB')
    image = results[1]
    image.save("./Users_pictures/users-"+actual_emotion+"-photo.jpg","JPEG")
    image.show()

#main("Happy") # Runs whole script
#Send a string with one of the emotions - to not have one emotion be tested
#Don't send it as an option.
