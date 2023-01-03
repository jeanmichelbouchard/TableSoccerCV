# TODO: error handling
# TODO: PEP8

#Copyright 2016 StudentCV
#Copyright and related rights are licensed under the
#Solderpad Hardware License, Version 0.51 (the “License”);
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at http://solderpad.org/licenses/SHL-0.51.
#Unless required by applicable law or agreed to in writing,
#software, hardware and materials distributed under this License
#is distributed on an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
#either express or implied. See the License for the specific language
#governing permissions and limitations under the License.


"""
This program analyses a tabletop soccer match utilizing a Basler dart camera to track the soccer ball.
Before the game starts, the system has to be calibrated to the ball color.
After the calibration it will find the field, track the ball and show the ball speed as well as
the score.

Throughout this program, the HSV representation of images will be used.
"""


import ball
import field
import imageSource
import GameplayAnalyser
import UserInterfaces

import cv2
import time


def calibrate_ball_color(ball_position):
    Interface.message("Please put the ball an the center spot!")

    # Interface.wait_for_user_command(Interface.start_calibration)

    Interface.message("Calibration starts - 5 seconds for positioning...")

    # The user has to put the ball onto the center spot for calibration.
    # A marker cross will appear on the center spot for some time.

    # At the moment the color calibration is done using a fixed image, that has
    # to be cropped to the right size.
    # Therefore, the size of the images from the camera is needed.
    t_end = time.time()  # + 1
    _done = 0
    # When the fixed image is used for calibration, at least one execution is
    # needed to get the size of the camera images
    while time.time() < t_end or not _done:

        image = Camera.getNewestFrame()
        x1 = int(round(ball_position[0] - image.shape[1]/20, 0))
        x2 = int(round(ball_position[0] + image.shape[1]/20, 0))
        y1 = int(round(ball_position[1] - image.shape[0]/20, 0))
        y2 = int(round(ball_position[1] + image.shape[0]/20, 0))

        marked_image = image.copy()
        # draw the marker cross
        cv2.line(marked_image, (x1, int(ball_position[1])), (x2, int(ball_position[1])), (0, 255, 255), 2)
        cv2.line(marked_image, (int(ball_position[0]), y1), (int(ball_position[0]), y2), (0, 255, 255), 2)

        Interface.show_video(marked_image, GetSourceVar)

        _done = 1

    # calibration_image = Camera.get_newest_frame()
    calibration_image = cv2.cvtColor(cv2.imread(r'./BallCalibrationJM.PNG'), cv2.COLOR_BGR2HSV)
    calibration_image = cv2.resize(calibration_image, (image.shape[1], image.shape[0]))

    ball.calibrate(image_crop)

#start = time.process_time() # --- program start --- #

# Initialize variables
Interface = UserInterfaces.PythonInterface()
ball = ball.BallTracker(Interface)
DetectField = field.field()
Camera = imageSource.imageSource()
Match = GameplayAnalyser.GameplayAnalyser()

Draw = [DetectField.draw, ball.draw, Match.draw]

GetSourceVar = Camera.getVar
GetFieldVar = DetectField.getVar
GetBallVar = ball.get_var
#GetMatchVar = Match.get_var

#  --- initialization --- #
#while not Interface.wait_for_user_command(Interface.start_session): #Ne fais rien actuellement
#    pass

#Interface.message("Session started")

Camera.startGrab()
#while not Camera.newImageAvailable():
#    pass
CalibrationImage = Camera.getNewestFrame()

# detection of the field
DetectField.calibrate(CalibrationImage)
# Match.SetField(field)

#calibrate_ball_color(DetectField.center)
image = Camera.getNewestFrame()
cv2.imshow("Before detect ball", image)
ball.calibrate(cv2.cvtColor(cv2.imread(r'./BallCalibrationJM.PNG'), cv2.COLOR_BGR2HSV), DetectField.center, DetectField.ratioPxCm)
ball.detect_ball_position(image)

#Interface.message("Calibration succeeded")

# --- match processing --- #
#Interface.wait_for_user_command(Interface.start_match)

#while Interface.run():
#    if not Camera.newImageAvailable():
#        continue

#    try:
#        Image = Camera.getNewestFrame()
#    except(cv2.error):
#        break
    
#    DetectBall.detect_ball_position(Image)
#    _ = cv2.waitKey(0)

#    GoalStatus = Match.check_for_goal(GetBallVar, GetFieldVar)
#    if GoalStatus is not False:
#        Match.count_score(GoalStatus)

#    Match.calc_ball_speed_average(GetSourceVar, GetBallVar, GetFieldVar, 10)

#    Match.heatmap(GetBallVar)

    # if(MatchMessage != "none"):
    #   Interface.Message(MatchMessage)

#    Interface.show_video(Image, GetSourceVar, Draw, )
    # DetectBall.Show() # for debugging

#print(time.process_time() - start)
