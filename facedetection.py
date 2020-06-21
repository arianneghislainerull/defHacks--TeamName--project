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

#classifier = load_model('./final_xception.h5')

cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cascade_eye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
cascade_smile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

global shape_x
global shape_y
global input_shape
global nClasses

def detection():
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

    model = load_model('video.h5')
    face_detect = dlib.get_frontal_face_detector()
    predictor_landmarks  = dlib.shape_predictor("face_landmarks.dat")

    vc = cv2.VideoCapture(0)

    #Call function in loop and get it to output guess

    while True:
        _, img = vc.read()
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = face_detect(grayscale,1)

        for (i, rect) in enumerate(rects):
            shape = predictor_landmarks(grayscale,rect)
            shape = face_utils.shape_to_np(shape)

            (x,y,w,h)= face_utils.rect_to_bb(rect) #Face coordinates
            face = grayscale[y:y+h,x:x+w]

            face = zoom(face, (shape_x / face.shape[0],shape_y / face.shape[1]))

            face = face.astype(np.float32)

            face /= float(face.max())
            face = np.reshape(face.flatten(), (1,48,48,1))

            emotionPrediction(face,model)

            # 3. Eye Detection and Blink Count
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            # Compute Eye Aspect Ratio
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # And plot its contours
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

            # 4. Detect Nose
            nose = shape[nStart:nEnd]
            noseHull = cv2.convexHull(nose)
            cv2.drawContours(img, [noseHull], -1, (0, 255, 0), 1)

            # 5. Detect Mouth
            mouth = shape[mStart:mEnd]
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(img, [mouthHull], -1, (0, 255, 0), 1)

            # 6. Detect Jaw
            jaw = shape[jStart:jEnd]
            jawHull = cv2.convexHull(jaw)
            cv2.drawContours(img, [jawHull], -1, (0, 255, 0), 1)

            # 7. Detect Eyebrows
            ebr = shape[ebrStart:ebrEnd]
            ebrHull = cv2.convexHull(ebr)
            cv2.drawContours(img, [ebrHull], -1, (0, 255, 0), 1)
            ebl = shape[eblStart:eblEnd]
            eblHull = cv2.convexHull(ebl)
            cv2.drawContours(img, [eblHull], -1, (0, 255, 0), 1)

        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vc.release()
    cv2.destroyAllWindows()


def emotionPrediction(face,model):

    emotion_prediction = model.predict(face)
    deciphered_emotion = predictionDecipher(emotion_prediction)

    predicted_class = np.argmax(emotion_prediction)
    #print(emotion_prediction)

    print(predicted_class)

def predictionDecipher(model_guess):
    anger_metric = str(round(model_guess[0][0],3)) #anger
    disgust_metric = str(round(model_guess[0][1],3)) #anger
    fear_metric = str(round(model_guess[0][2],3)) #anger
    happy_metric = str(round(model_guess[0][3],3)) #anger
    sad_metric = str(round(model_guess[0][4],3)) #anger
    surprise_metric = str(round(model_guess[0][5],3)) #anger
    neutral_metric = str(round(model_guess[0][6],3)) #anger
    print("Anger:"+anger_metric)
    print("Disgust:"+disgust_metric)
    print("Fear:"+fear_metric)
    print("Happy:"+happy_metric)
    print("Sad:"+sad_metric)
    print("Surprise:"+surprise_metric)
    print("Neutral:"+neutral_metric)

detection()
