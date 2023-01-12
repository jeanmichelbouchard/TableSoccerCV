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

#backSubMOG = cv2.createBackgroundSubtractorMOG()
backSubMOG2 = cv2.createBackgroundSubtractorMOG2()
backSubKNN = cv2.createBackgroundSubtractorKNN()

#backSub.nmixtures = 3
backSubMOG2.setDetectShadows(True)
backSubMOG2.setShadowValue(0)
#backSub.nShadowDetection = 0
#backSub.fTau = 0.5

# detection of the field borders
field.calibrate(fieldCalibrationImage, verbose=1)

ballCalibrationImage = cv2.cvtColor(cv2.imread(r'./media/BallCalibration_liege.PNG'), cv2.COLOR_RGB2HSV)
ballCalibrationImage = cv2.resize(ballCalibrationImage, (fieldCalibrationImage.shape[1], fieldCalibrationImage.shape[0]))
ball.calibrateMethod2(ballCalibrationImage, field.center, field.ratioPxCm, verbose=0)

frameCount = 0
frame = imageSource.getNewestFrame()

while imageSource.newImageAvailable():

    fgMaskMOG2 = backSubMOG2.apply(frame)
    fgMaskKNN = backSubKNN.apply(frame)
    cv2.imshow("Frame", frame)
    cv2.imshow("fgMaskMOG2", fgMaskMOG2)
    cv2.imshow("fgMaskKNN", fgMaskKNN)

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
