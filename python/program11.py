# Advanced face detection and tracking

import cv2    #opencv library
alg = "haarcascade_frontalface_default.xml"     # machine learing algorithm fro traine the image footage to pick the face detection
haar_cascade = cv2.CascadeClassifier(alg)    # using the algorithm in that above xml file folder

# using already image footage
video_path = "DK.jpg"    # image footage
cam = cv2.VideoCapture(video_path)     

while True:
    _,img = cam.read()     #reading the frame from Camera
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    # converting the CLR image to gray
    face = haar_cascade.detectMultiScale(grayImg,1.3,4)      #getting colour coordination
    for(x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow("Facedetection",img)

    key = cv2.waitKey(10)
    if key == 27:
        break
cam.release()
cv2.destroywindows()
