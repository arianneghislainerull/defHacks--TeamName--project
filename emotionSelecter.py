import random
from PIL import Image
from pathlib import Path
from playsound import playsound

# I just found one picture for each of the emotions that is used in a emotion
# recognition bot I found. Finding more will probably be needed

# this script return an image for an emotion or can randomly select an emotion


# Image paths
happyImagePath = Path("./Faces/happy.jpg")
sadImagePath = Path("./Faces/sad.jpg")
angryImagePath = Path("./Faces/angry.jpg")
neutralImagePath = Path("./Faces/neutral.jpg")
disgustImagePath = Path("./Faces/disgust.jpg")
surpriseImagePath = Path("./Faces/surprise.jpg")
fearImagePath = Path("./Faces/fear.jpg")

# Sound paths (to make each sound monotoned add "-mono" after the emotion)
happySoundPath = "./Sounds/happy.mp3"
sadSoundPath = "./Sounds/sad.mp3"
angrySoundPath = "./Sounds/angry.mp3"
neutralSoundPath = "./Sounds/neutral.mp3"
disgustSoundPath = "./Sounds/disgust.mp3"
surpriseSoundPath = "./Sounds/surprise.mp3"
fearSoundPath = "./Sounds/fear.mp3"


#Creating and resizing the images
happyImage = Image.open(happyImagePath)
happyImage = happyImage.resize((512,512))

sadImage = Image.open(sadImagePath)
sadImage = sadImage.resize((512,512))

angryImage = Image.open(angryImagePath)
angryImage = angryImage.resize((512,512))

neutralImage = Image.open(neutralImagePath)
neutralImage = neutralImage.resize((512,512))

disgustImage = Image.open(disgustImagePath)
disgustImage = disgustImage.resize((512,512))

surpriseImage = Image.open(surpriseImagePath)
surpriseImage = surpriseImage.resize((512,512))

fearImage = Image.open(fearImagePath)
fearImage = fearImage.resize((512,512))


def randomEmotion():
    emotionNumber = random.randint(1,7)
    print(emotionNumber)
    if(emotionNumber == 1):
        return angry()
    elif(emotionNumber == 2):
        return disgust()
    elif(emotionNumber == 3):
        return fear()
    elif(emotionNumber == 4):
        return happy()
    elif(emotionNumber == 5):
        return neutral()
    elif(emotionNumber == 6):
        return sad()
    elif(emotionNumber == 7):
        return surprise()
    else:
        return "Error"

#Each function returns a photo based on the emotion selected

# Could also verbally say the word

def happy():
    happyImage.show()
    playsound(happySoundPath)
    return happyImage

def angry():
    angryImage.show()
    playsound(angrySoundPath)
    return angryImage

def sad():
    sadImage.show()
    playsound(sadSoundPath)
    return sadImage

def neutral():
    neutralImage.show()
    playsound(neutralSoundPath)
    return neutralImage

def disgust():
    disgustImage.show()
    playsound(disgustSoundPath)
    return disgustImage

def surprise():
    surpriseImage.show()
    playsound(surpriseSoundPath)
    return surpriseImage

def fear():
    fearImage.show()
    playsound(fearSoundPath)
    return fearImage

randomEmotion()
