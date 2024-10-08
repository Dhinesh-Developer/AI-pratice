import cv2
image = cv2.imread("LT.jpg")
cv2.imshow('Logo',image)
cv2.imwrite('image1.png',img)
cv2.waitKey(10000)
cv2.destroyAllWindows()
