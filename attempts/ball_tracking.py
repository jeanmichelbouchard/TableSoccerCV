import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
cap = cv2.VideoCapture('1647562380_replay_short.h264')
#whiteLower = np.array([0,0,168], dtype='uint8')
#whiteUpper = np.array([172,111,255], dtype='uint8')

whiteHSV =   np.array([29, 13, 222], dtype='uint8')
whiteLower = np.array([19, 100, 100], dtype='uint8') # - 10, 100, 100
whiteUpper = np.array([39, 255, 255], dtype='uint8') # + 10, 255, 255

c = 0
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 

print('frame_width: ' + str(frame_width))
print('frame_height: ' + str(frame_height))

size = (frame_width, frame_height) 

result = cv2.VideoWriter('balltracking.mp4',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size) 
while True:
    grapped,frame=cap.read()
    if grapped == True:
        
        white = cv2.inRange(frame,whiteLower,whiteUpper)
        white = cv2.GaussianBlur(white,(3,3),0)

        cnts = cv2.findContours(white.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if len(cnts) > 0:
            cnt = sorted(cnts,key=cv2.contourArea,reverse=True)[0]
            rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
            cv2.circle(frame, (rect[0][0]+(rect[-1][0] - rect[0][0])//2,rect[1][1]+(rect[-1][-1]-rect[1][1])//2), 
                   25, (0, 255, 0), -1)
        cv2.imshow("Ball Tracking", frame)
        result.write(frame)
        if cv2.waitKey() & 0xFF == ord("q"):
            break
        
    else:
        break
        

# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()