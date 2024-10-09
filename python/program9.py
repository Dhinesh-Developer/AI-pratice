# find contours
# syntax = cv2.findContours(srcImage.copy(),contourRetrievelMode,contourApproximationMethod)

cnts = cv2.findContours(threshImg.copy(),cv2.RETR_EXTERNAL<CV2.CHAIN_APPROX_SIMPLE)


# Drawing Rectangle
#cv2.rectangle(src,startpoint,endpoint,color,thickness)

cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


#putting text in Image
#cv2.putText(src,text,position,font,fontsize,color,thickness)

cv2.puttext(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
