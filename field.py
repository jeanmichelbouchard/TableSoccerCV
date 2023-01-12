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

# TODO: extensive commenting on calculations
import cv2
import numpy as np
import math

class field:
    """
    The soccer field is determined by the position of the center spot, the
    angle of the center line and the size of the center circle.
    Since the diameter of the center circle is fixed at 20.5 cm, all other
    points of the field can be calculated by these three measures.
    """

    field = 0
    center = 0 # [x, y, radius of center circle]
    ratioPxCm = 0
    angle = 0
    goalCenterLeft = 0
    goalCenterRight = 0
    goalAreaRadius = 0

    def calibrate(self, calibrationImage, verbose=0):
        calibrationImageHSV = cv2.cvtColor(calibrationImage, cv2.COLOR_RGB2HSV)
        self._getCenterScale(calibrationImageHSV, verbose)
        self._getAngle(calibrationImageHSV, verbose)
        self._calcField()
        self._calcGoalArea()

    def inField(self, position):
        """
        :param position: Coordinates of a point
        :return: True if the position is inside the field, False otherwise
        """
        verbose = 0

        # Field has not been calibrated yet
        if self.field == 0:
            return False

        topLeft = self.field[0]
        bottomRight = self.field[2]
    
        if (topLeft[0] <= position[0] <= bottomRight[0] and topLeft[1] <= position[1] <= bottomRight[1]) :
            return True
        else :
            if verbose:
                print(str(position) + " is not between" + str(topLeft) + "and " + str(bottomRight))
            return False

    def draw(self, image):
        """
        Draws the field borders and markers onto the image.
        :param image: The HSV-image to draw on
        :return: The image with the markers drawn
        """
        imageCopy = image.copy()

        if self.field != 0:
            topLeft = self.field[0]
            topRight = self.field[1]
            bottomRight = self.field[2]
            bottomLeft = self.field[3]
            goalCenterRight = self.goalCenterRight
            goalCenterLeft = self.goalCenterLeft

            # Draw center circle
            cv2.circle(imageCopy, (int(self.center[0]), int(self.center[1])), int(self.center[2]), (0, 255, 0), 1)

            # Draw the center marker cross
            x1 = int(round(self.center[0] - imageCopy.shape[1]/20, 0))
            x2 = int(round(self.center[0] + imageCopy.shape[1]/20, 0))
            y1 = int(round(self.center[1] - imageCopy.shape[0]/20, 0))
            y2 = int(round(self.center[1] + imageCopy.shape[0]/20, 0))
            cv2.line(imageCopy, (x1, int(self.center[1])), (x2, int(self.center[1])), (0, 255, 255), 2)
            cv2.line(imageCopy, (int(self.center[0]), y1), (int(self.center[0]), y2), (0, 255, 255), 2)

            # Draw the field lines
            cv2.line(imageCopy, (topLeft[0], topLeft[1]), (topRight[0], topRight[1]), (120, 255, 255), 2)
            cv2.line(imageCopy, (topRight[0], topRight[1]), (bottomRight[0], bottomRight[1]), (120, 255, 255), 2)
            cv2.line(imageCopy, (bottomRight[0], bottomRight[1]), (bottomLeft[0], bottomLeft[1]), (120, 255, 255), 2)
            cv2.line(imageCopy, (bottomLeft[0], bottomLeft[1]), (topLeft[0], topLeft[1]), (120, 255, 255), 2)

            # Draw the goal areas
            cv2.circle(imageCopy, (goalCenterLeft[0], goalCenterLeft[1]), self.goalAreaRadius, (120, 255, 255), 2)
            cv2.circle(imageCopy, (goalCenterRight[0], goalCenterRight[1]), self.goalAreaRadius, (120, 255, 255), 2)

        else:
            print('Field was not determined!')  # TODO: Interface message or error code in return value

        return imageCopy

    def _getCenterScale(self, calibrationImage, verbose=0):
        """
        :param calibrationImage: The HSV-image to use for calculation
        :return: Position of center point in image (tuple), ratio px per cm (reproduction scale)
        """
        # Some preprocessing
        rgb = cv2.cvtColor(calibrationImage, cv2.COLOR_HSV2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)
        thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,2)
        if verbose:
            cv2.imshow("adaptiveThreshold", thresh)
            cv2.waitKey(0)

        # Detect circles in image
        circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=30, minRadius=50, maxRadius=300)

        # Select the nearest circle to the center of calibrationImage
        centerCircle = (0, 0, 0)
        minDistance = 0xFFFFFFFFFFF

        for circle in circles[0]:
            distX = abs(circle[0] - calibrationImage.shape[1] / 2)
            distY = abs(circle[1] - calibrationImage.shape[0] / 2)
            if(distX + distY) < minDistance:
                minDistance = distX + distY
                centerCircle = circle
                # Show each candidate circle in red
                if verbose == 1:
                    cv2.circle(blurred, (int(centerCircle[0]), int(centerCircle[1])), int(centerCircle[2]), (255, 0, 0), 1)
                    cv2.imshow("Center circle", blurred)

        # Draw chosen circle in green
        if verbose == 1:
            cv2.circle(blurred, (int(centerCircle[0]), int(centerCircle[1])), int(centerCircle[2]), (0, 255, 0), 1)
            cv2.imshow("Center circle", blurred)

        # Since the diameter of the center circle is fixed at 20.5 cm, we can calculate the ratio of 
        # pixels / cm from the radius of the center circle found.
        radius = centerCircle[2]
        ratioPxCm = radius / 8.75 # Normally 10.25 but trying other values for my table

        self.center = centerCircle
        self.ratioPxCm = ratioPxCm

        print("Center circle: " + str(centerCircle) + " ratioPxCm: " + str(ratioPxCm))

        return [self.center, self.ratioPxCm]

    def _distanceToLine(self, x0,y0, x1, y1, x2 ,y2):
        """
        :param (x0,y0): point in space
        :param (x1,y1), (x2,y2): two points that define a line to calculate the distance from
        :return: Distance from point to line
        """
        x_diff = x2 - x1
        y_diff = y2 - y1
        num = abs(y_diff*x0 - x_diff*y0 + x2*y1 - y2*x1)
        den = math.sqrt(y_diff**2 + x_diff**2)
        return num / den

    def _getAngle(self, calibrationImage, verbose=0):
        """
        Method _getCenterScale has to be called before.
        :param calibrationImage: The HSV-image to use for calculation
        :return: Rotation angle of the field in image
        """
        angle = 0
        
        # The initialization is done with an square of 4x the radius of the center circle at the center of the field.
        # This way we maximize the chances to detect the center line first.
        x1 = int(round(self.center[0] - 2*self.center[2], 0))
        x2 = int(round(self.center[0] + 2*self.center[2], 0))
        y1 = int(round(self.center[1] - 2*self.center[2], 0))
        y2 = int(round(self.center[1] + 2*self.center[2], 0))
        print("(x1,x2): " + str(((x1,x2))) + "(y1,y2): " + str(((y1,y2))))
        cropped = calibrationImage[y1:y2, x1:x2]
        cv2.imshow("_getAngle : Center line", cropped)
        cv2.waitKey(0)

        rgb   = cv2.cvtColor(cropped, cv2.COLOR_HSV2RGB)
        gray  = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines in image
        # https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
        lines = cv2.HoughLines(edges, 1, np.pi/180, 110)

        if lines.shape[0]:
            line_count = lines.shape[0]
        else:
            raise Exception('field not detected')

        # Take the first line as OpenCv will return the lines in order of their confidence, with the strongest line first. It should be the field center line.
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*a)
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*a)
            angle = np.degrees(b)

        # Show the frame to our screen
        if verbose == 1:
            cv2.line(rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow("_getAngle : Center line", rgb)
            cv2.imshow("_getAngle : gray", gray)
            cv2.imshow("_getAngle : edges", edges)
            print('Detected table angle: ' + str(angle))

        self.angle = angle
        return angle

    def _calcField(self):
        """
        This method needs some class variables. get_angle and get_center_scale have to be called beforehand.
        :return: field edges [Top left, top right, bottom right and bottom left corner] (list)
        """

        half_field_width_cm = 60
        half_field_height_cm = 34
        half_field_width_px = half_field_width_cm * self.ratioPxCm
        half_field_height_px = half_field_height_cm * self.ratioPxCm

        angle_radial_scale = np.radians(self.angle)

        x2 = int(self.center[0] - half_field_width_px + np.tan(angle_radial_scale) * half_field_height_px)
        y2 = int(self.center[1] - half_field_height_px - np.tan(angle_radial_scale) * half_field_width_px)
        topLeft = [x2, y2]

        x2 = int(self.center[0] + half_field_width_px + np.tan(angle_radial_scale) * half_field_height_px)
        y2 = int(self.center[1] - half_field_height_px + np.tan(angle_radial_scale) * half_field_width_px)
        topRight = [x2, y2]

        x2 = int(self.center[0] - half_field_width_px - np.tan(angle_radial_scale) * half_field_height_px)
        y2 = int(self.center[1] + half_field_height_px - np.tan(angle_radial_scale) * half_field_width_px)
        bottomLeft = [x2, y2]

        x2 = int(self.center[0] + half_field_width_px - np.tan(angle_radial_scale) * half_field_height_px)
        y2 = int(self.center[1] + half_field_height_px + np.tan(angle_radial_scale) * half_field_width_px)
        bottomRight = [x2, y2]

        self.field = [topLeft, topRight, bottomRight, bottomLeft]
        return [topLeft, topRight, bottomRight, bottomLeft]

    def _calcGoalArea(self):
        """
        The 'goal area' is the half circle around the goals. It is assumed,
        that the ball will be seen in a goal area before the score is
        incremented.
        :return: None
        """
        topLeft = self.field[0]
        topRight = self.field[1]
        bottomLeft = self.field[3]

        tlbl = bottomLeft[0] - topLeft[0], bottomLeft[1] - topLeft[1]  # Topleft to BottomLeft
        tltr = topRight[0] - topLeft[0], topRight[1] - topLeft[1]  # Topleft to TopRight

        self.goalCenterLeft = int(topLeft[0] + (0.5 * tlbl[0])), int(topLeft[1] + (0.5 * tlbl[1]))
        self.goalCenterRight = int(self.goalCenterLeft[0] + tltr[0]), int(self.goalCenterLeft[1] + tltr[1])

        self.goalAreaRadius = int(12 * self.ratioPxCm)

    def getVar(self, _type):
        """
        Get the class variables
        :param _type: String to choose the variabe
        :return: The requested variable, empty string if requested name is
        unavailable
        """
        if 'GoalAreas' == _type:
            return [self.goalCenterLeft, self.goalCenterRight, self.goalAreaRadius]
        elif 'field' == _type:
            return self.field
        elif 'ratioPxCm' == _type:
            return self.ratioPxCm
        elif 'angle' == _type:
            return self.angle
        elif 'center' == _type:
            return self.center
        else:
            return ""  # False
