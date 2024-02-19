import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img) 
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        cropImg = img[y-offset: y + h + offset, x-offset: x + w + offset]
        whiteImg = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        imgCropShape = cropImg.shape
        aspectRatio = h / w
        
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(cropImg, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            whiteImg[:, wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(whiteImg, draw=False)
            print(prediction, index)
        
        else:
            k = imgSize / w
            h_cal = math.ceil(k * h)
            img_resize = cv2.resize(cropImg, (imgSize, h_cal))
            img_resize_shape = img_resize.shape
            h_gap = math.ceil((imgSize - h_cal) / 2)
            whiteImg[h_gap:h_cal + h_gap, :] = img_resize
            prediction, index = classifier.getPrediction(whiteImg, draw=False)
        
        cv2.rectangle(imgOutput, (x-offset, y-offset-90), (x+offset+70, y+offset-90+50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y-40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)
        
        cv2.imshow("CropImage", cropImg)
        cv2.imshow("White Img", whiteImg)

        
    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    
    if key == ord("q") or key == ord("Q"):
        break
    