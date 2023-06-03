import cv2


# Create our body classifier

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while (True):
    
    # Read first frame
    ret, frame = cap.read()
    

    #Convert Each Frame into Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    body_cas = cv2.CascadeClassifier("C:/Python310/Lib/site-packages/cv2/data/haarcascade_fullbody.xml")
    # Pass frame to our body classifier
    bodie = body_cas.detectMultiScale(gray,1.2,4)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in bodie:
       cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                 
    # Display the resulting frame
    cv2.imshow(cap,frame)
      

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()

cv2.destroyAllWindows()
