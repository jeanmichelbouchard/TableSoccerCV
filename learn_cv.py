import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

import imageSource
import field
import ball
import UserInterfaces

# Initialize variables
imageSource = imageSource.imageSource(False, 'media/1673126962_replay_short.h264') # Balle liège
field = field.field()
ball = ball.ball()

fieldCalibrationImage = imageSource.getNewestFrame()

# detection of the field
field.calibrate(fieldCalibrationImage)

ballCalibrationImage = cv2.cvtColor(cv2.imread(r'./media/BallCalibration_liege.PNG'), cv2.COLOR_RGB2HSV)
ballCalibrationImage = cv2.resize(ballCalibrationImage, (fieldCalibrationImage.shape[1], fieldCalibrationImage.shape[0]))
ball.calibrateMethod2(ballCalibrationImage, field.center, field.ratioPxCm, verbose=1)

def nothing(x):
    pass

cv2.namedWindow('temp')
cv2.createTrackbar('Hl', 'temp', 101, 255, nothing)
cv2.createTrackbar('Sl', 'temp', 88, 255, nothing)
cv2.createTrackbar('Vl', 'temp', 0, 255, nothing)
cv2.createTrackbar('Hh', 'temp', 116, 255, nothing)
cv2.createTrackbar('Sh', 'temp', 158, 255, nothing)
cv2.createTrackbar('Vh', 'temp', 109, 255, nothing)
cv2.createTrackbar('Radius', 'temp', 2, 10, nothing)
cv2.createTrackbar('Circle', 'temp', 2, 10, nothing)

frameCount = 0
frame = imageSource.getNewestFrame()

while imageSource.newImageAvailable():

#    ball.findMethod0(frame.copy())
    ball.findMethod20(field, frame.copy(), verbose=1)
#    img = field.draw(frame.copy())

    key = cv2.waitKeyEx(0)

    # Quit
    if key == ord('q'):
        break
    # Save frame to file
    elif key == ord('s'):
        cv2.imwrite(r'./frame_' + str(frameCount) + '.png', cv2.cvtColor(frame, cv2.COLOR_HSV2RGB))
    # Right arrow key: Get next frame
    elif key == 2555904 :
        try:
            frame = imageSource.getNewestFrame()
            frameCount = frameCount + 1
        except(cv2.error):
            break

"""
Ball HSV color detected: (92.77998017839445, 148.11595639246778, 59.02477700693756)
82-102
100-200
9-109

gray = cv2.cvtColor(calibration_image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
edged = cv2.Canny(gray, 10, 100)
# find the contours in the dilated image
contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
image_copy = calibration_image.copy()
# draw the contours on a copy of the original image
cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
cv2.imshow("Edged", edged)
cv2.imshow("Contours", image_copy)
cv2.waitKey(0)
"""

#destroys all window
cv2.destroyAllWindows()