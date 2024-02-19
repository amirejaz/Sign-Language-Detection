import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = "Data/E"
counter = 0

while True:
    success, img = cap.read()
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
        
        else:
            k = imgSize / w
            h_cal = math.ceil(k * h)
            img_resize = cv2.resize(cropImg, (imgSize, h_cal))
            img_resize_shape = img_resize.shape
            h_gap = math.ceil((imgSize - h_cal) / 2)
            whiteImg[h_gap:h_cal + h_gap, :] = img_resize
        
        cv2.imshow("CropImage", cropImg)
        cv2.imshow("White Img", whiteImg)

        
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    
    if key == ord("q") or key == ord("Q"):
        break
    if key == ord("s") or key == ord("S"):
        counter += 1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", whiteImg)
        print(counter)