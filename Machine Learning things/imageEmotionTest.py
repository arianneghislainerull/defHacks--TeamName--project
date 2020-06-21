from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import face_recognition
import keras
from keras.models import load_model
import cv2

emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}


face_image  = cv2.imread("./test_images/disgust.jpg")
plt.imshow(face_image)

face_image = cv2.resize(face_image, (48,48), fx=0.25, fy=0.25)
face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])


model = load_model("./model_v6_23.hdf5")
print("Trained:")
print (face_image.shape) #Trained prediction

predicted_class = np.argmax(model.predict(face_image))

label_map = dict((v,k) for k,v in emotion_dict.items())
predicted_label = label_map[predicted_class]

print(predicted_label)
