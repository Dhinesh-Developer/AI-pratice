 # Reading frame from camera-video Streaming

import cv2
vs = cv2.VideoCapture(0)   # intializing camera ID
while True:
     _,img = vs.read();
     cv2.imshow("VideoStreaming",img)
     key = cv2.waitKey(1)
     print(key)
     if key == ord("d"):
         break;
vs.release()
cv2.destroyAllWindows()
