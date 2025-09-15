import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3

# 🎤 Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)   # speed of speech
engine.setProperty('volume', 1)   # volume (0.0 to 1.0)

# 🎥 OpenCV + Hand Detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# 🧠 Load your trained model
classifier = Classifier(
    r"C:\Users\Advance solution\Desktop\converted_keras\keras_model.h5",
    r"C:\Users\Advance solution\Desktop\converted_keras\labels.txt"
)

offset = 20
imgSize = 300

# Define your labels (must match labels.txt order)
labels = ["Hello", "I love you", "Yes", "Thank you"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]

        try:
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            # 🧠 Prediction from local model
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            label = labels[index]

            # 🎤 Speak immediately on every detection
            engine.say(label)
            engine.runAndWait()

            # 🖼️ Draw results
            cv2.rectangle(imgOutput, (x-offset, y-offset-70),
                          (x-offset+400, y-offset+60-50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, label, (x, y-30),
                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset),
                          (x + w + offset, y + h + offset), (0, 255, 0), 4)

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

        except Exception as e:
            print("Error:", e)

    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)

