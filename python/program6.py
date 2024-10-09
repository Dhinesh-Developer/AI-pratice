# Threshold value

import cv2
img = cv2.imread("LT.jpg")
greyImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresholdImg = cv2.threshold(greyImg,180,155,cv2.THRESH_BINARY)[1]
cv2.imshow("original",img)
cv2.imshow("thresh",thresholdImg)
