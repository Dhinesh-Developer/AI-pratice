import urllib.request
import cv2
import imutils
import numpy as np
import ssl

# Disabling SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

url = "https://192.168.161.123:8080/shot.jpg"
while True:
    # Open the URL and read the image
    imgPath = urllib.request.urlopen(url)
    imgNP = np.array(bytearray(imgPath.read()), dtype=np.uint8)
    
    # Decode the image
    frame = cv2.imdecode(imgNP, -1)
    
    # Resize the frame
    frame = imutils.resize(frame, width=500)
    
    # Display the frame
    cv2.imshow("Frame", frame)
    
    # Exit the loop if 'q' is pressed
    if ord("q") == cv2.waitKey(1):
        break

# Clean up windows
cv2.destroyAllWindows()
