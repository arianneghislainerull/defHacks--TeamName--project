import random
from PIL import Image
from pathlib import Path
from playsound import playsound

def thepath(emotion):
    #randomly choose a path of the given expression
    list = ["./Faces/"+emotion+".jpg", "./Faces/"+emotion+"2.jpg", "./Faces/"+emotion+"3.jpg"]
    chosenpath = random.choice(list)
    return chosenpath


def soundpath(emotion):
    return "./Sounds/"+emotion+".mp3"


#randomly select a picture of the given emotion and paly the sound
#emotion is a string eg. reaction("happy")
def reaction(emotion):
    image = Image.open(Path(thepath(emotion)))
    #resize Image
    image = image.resize((512,512))
    #image.show()
    #playsound(soundpath(emotion))
    return image , soundpath(emotion)

def randomEmotion():
    #choose randomly from all emotions
    list = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
    emotion = random.choice(list)
    return reaction(emotion) , emotion


#randomEmotion() Uncomment to test
