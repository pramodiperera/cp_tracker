import numpy as np
import cv2
import cvzone
import pickle

cap = cv2.VideoCapture('carPark.mp4')
width, height = 107, 48
with open('parkingPositions', 'rb') as f:
    posList = pickle.load(f)


def checkParkingSpace(imgPrecessed):

    spaceCounter = 0

    for pos in posList:
        x, y = pos

        imgCrop = imgPrecessed[y:y+height, x:x+width]
        count = cv2.countNonZero(imgCrop)
        cvzone.putTextRect(img, str(count), (x, y+height-3), scale=1, thickness=2, offset=0, colorR=(0, 0, 255))

        if count < 900:
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)

    cvzone.putTextRect(img, f'Free:{spaceCounter}/{len(posList)}', (100, 50), scale=2, thickness=4, offset=20, colorR=(0, 200, 0))


while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgTreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgTreshold, 5)
    kernal = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernal, iterations=1)

    checkParkingSpace(imgDilate)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

