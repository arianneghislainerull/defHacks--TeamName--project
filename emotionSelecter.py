import random
from PIL import Image
from pathlib import Path

happyPath = Path("./Faces/happy.jpg")
sadPath = Path("./Faces/sad.jpg")
angryPath = Path("./Faces/angry.jpg")
neutralPath = Path("./Faces/neutral.jpg")
disgustPath = Path("./Faces/disgust.jpg")
surprisePath = Path("./Faces/surprise.jpg")
fearPath = Path("./Faces/fear.jpg")

happyImage = Image.open(happyPath)
happyImage = happyImage.resize((512,512))

sadImage = Image.open(sadPath)
sadImage = sadImage.resize((512,512))

angryImage = Image.open(angryPath)
angryImage = angryImage.resize((512,512))

neutralImage = Image.open(neutralPath)
neutralImage = neutralImage.resize((512,512))

disgustImage = Image.open(disgustPath)
disgustImage = disgustImage.resize((512,512))

surpriseImage = Image.open(surprisePath)
surpriseImage = surpriseImage.resize((512,512))

fearImage = Image.open(fearPath)
fearImage = fearImage.resize((512,512))

def randomEmotion():
    emotionNumber = random.randint(1,7)
    switcher = {
        1: angry(),
        2: disgust(),
        3: fear(),
        4: happy(),
        5: neutral(),
        6: sad(),
        7: surprise()
    }
    return switcher.get(emotionNumber,"Error")

def happy():
    #happyImage.show()
    return happyImage

def angry():
    #angryImage.show()
    return angryImage

def sad():
    #sadImage.show()
    return sadImage

def neutral():
    #neutralImage.show()
    return neutralImage

def disgust():
    #disgustImage.show()
    return disgustImage

def surprise():
    #surpriseImage.show()
    return surpriseImage

def fear():
    #fearImage.show()
    return fearImage

#randomEmotion()
