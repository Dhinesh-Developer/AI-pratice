# resize an image.

import cv2
import imutils   # imutils library ,install it and use
img = cv2.imread('LT.jpg')
resizedImg = imutils.resize(img,width=100)   # resize the image syntax
cv2.imshow('original.jpg',img)
cv2.imshow('Resized Image',resizedImg)
