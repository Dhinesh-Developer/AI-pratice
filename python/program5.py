# Gaussian Blur - Smootheneing

import cv2      # importing libraries
import imutils  # importing libraries
img = cv2.imread('LT.jpg')  # fetching the image from folder
img = imutils.resize(img,width=200)
gaussianImg = cv2.GaussianBlur(img,(21,21),2)   # perfect mild smoothing
cv2.imshow("original.jpg",img)
cv2.imshow("ModifiesBlur",gaussianImg)
cv2.imwrite("Blur.jpg",gaussianImg)   # storing the image in new format

gaussianImg1 = cv2.GaussianBlur(img,(41,41),2)   # for reference not use this
cv2.imshow("ModifiesBlur",gaussianImg1)
