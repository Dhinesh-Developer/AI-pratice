import cv2  # importing the cv2 libraray
image = cv2.imread("LT.jpg")   # read the image and store it in new variable
cv2.imshow('original',image)     # to display the original image
grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)     #changing the image color to grey color
cv2.imshow('modified',image)    # to display the modifies image
print(image.shape)    #properties to showing how many column,rows,colour used 
print(image.size)     # properties to showing the image size just multiply with rows and columns
