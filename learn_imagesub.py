import cv2
import numpy as np
import matplotlib.pyplot as plt

import imageSource
import field
import ball

# Initialize variables
imageSource = imageSource.imageSource(False, 'media/1647562380_replay_short.h264') # Balle blanche
#imageSource = imageSource.imageSource(False, 'media/1673126962_replay_short.h264') # Balle liège
#imageSource = imageSource.imageSource(False, 'media/90fps.h264')

field = field.field()
ball = ball.ball()

fieldCalibrationImage = imageSource.getNewestFrame()

backSub = cv2.createBackgroundSubtractorMOG2()
#backSub = cv2.createBackgroundSubtractorKNN()

# detection of the field borders
field.calibrate(fieldCalibrationImage, verbose=1)

ballCalibrationImage = cv2.cvtColor(cv2.imread(r'./media/BallCalibration_liege.PNG'), cv2.COLOR_RGB2HSV)
ballCalibrationImage = cv2.resize(ballCalibrationImage, (fieldCalibrationImage.shape[1], fieldCalibrationImage.shape[0]))
ball.calibrateMethod2(ballCalibrationImage, field.center, field.ratioPxCm, verbose=0)

frameCount = 0
frame = imageSource.getNewestFrame()

fieldNoBall = cv2.cvtColor(cv2.imread(r'./media/foosballFieldNoBall.png'), cv2.COLOR_BGR2HSV)
fieldNoBall = cv2.resize(fieldNoBall, (frame.shape[1], frame.shape[0]))

while imageSource.newImageAvailable():

    substract = cv2.subtract(fieldNoBall, frame)
    cv2.imshow("Frame", frame)
    cv2.imshow("substract", substract)

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
