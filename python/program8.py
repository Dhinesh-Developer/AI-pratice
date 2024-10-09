# Threshold value

import cv2
img = cv2.imread("LT.jpg")
grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# syntax = cv2.threshold(src,threshold,maxValueForThreshold,binary,type)[1]

threshImg = cv2.threshold(grayImg,180,255,cv2.THRESH_BINARY)[1]
cv2.imshow("original",img)
cv2.imshow("threshImage",threshImg)
