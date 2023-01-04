import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

import imageSource
import field
import ball
import UserInterfaces

# Initialize variables
imageSource = imageSource.imageSource()
field = field.field()
ball = ball.ball()

fieldCalibrationImage = imageSource.getNewestFrame()

# detection of the field
field.calibrate(fieldCalibrationImage)
#img = field.draw(calibration_image.copy())

ballCalibrationImage = cv2.cvtColor(cv2.imread(r'./media/BallCalibrationJM.PNG'), cv2.COLOR_RGB2HSV)
ballCalibrationImage = cv2.resize(ballCalibrationImage, (fieldCalibrationImage.shape[1], fieldCalibrationImage.shape[0]))
ball.calibrate(ballCalibrationImage, field.center, field.ratioPxCm)

frameCount = 0
while imageSource.newImageAvailable():
    try:
        frame = imageSource.getNewestFrame()
        frameCount = frameCount + 1
    except(cv2.error):
        break

    ball.detectBallPosition(frame)

    key = cv2.waitKey(0)
    # Quit
    if key == ord('q'):
        break
    # Save frame to file
    elif key == ord('s'):
        cv2.imwrite(r'./frame_' + str(frameCount) + '.png', cv2.cvtColor(frame, cv2.COLOR_HSV2RGB))


"""

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
