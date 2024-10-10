#oject tracking based on colour with openCV.


import imutils
import cv2

# Define the lower and upper bounds for the red color in HSV space
redLower = (95, 49, 100)
redUpper = (154, 255, 255)

# Initialize the camera
camera = cv2.VideoCapture(0)  # Camera ID

while True:
    (grabbed, frame) = camera.read()  # Grab the current frame
    frame = imutils.resize(frame, width=1000)  # Resize the frame
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)  # Blur the frame
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # Convert to HSV color space

    # Create a mask for the red color
    mask = cv2.inRange(hsv, redLower, redUpper)
    mask = cv2.erode(mask, None, iterations=2)  # Erode the mask to remove noise
    mask = cv2.dilate(mask, None, iterations=2)  # Dilate the mask to restore size

    # Find contours in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # If at least one contour is found
    if len(cnts) > 0:
        # Find the largest contour
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        # Calculate the center of the contour
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # If the radius is above a certain threshold
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)  # Draw the circle
            cv2.circle(frame, center, 5, (0, 0, 255), -1)  # Draw the center point

            # Print control commands based on the center's position
            if radius > 250:
                print("Stop")
            else:
                if center[0] > 450:
                    print("Right")
                elif center[0] < 150:
                    print("Left")
                elif center[0] < 250:
                    print("Front")
                else:
                    print("Stop")

    # Show the frame
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()
