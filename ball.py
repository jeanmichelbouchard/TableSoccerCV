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

import cv2
import numpy as np

class ball:
    # --- ball detection in a single image --- #

    ball_detection_threshold = 0.2

    ball_color = (-1, -1, -1)
    ball_radius = 0
    curr_ball_position = (-1, -1)

    def __init__(self):
        super().__init__

    def draw(self, image):
        """
        Draws the ball marker onto the image.
        :param image: The image to draw on
        :return: Image with the marker drawn
        """
        imageCopy = image.copy()
        if self.curr_ball_position != (-1, -1):
            cv2.circle(imageCopy, (self.curr_ball_position[0], self.curr_ball_position[1]), self.ball_radius, (0, 255, 255), 1)
        else:
            # TODO : return error or message
            pass
        
        return imageCopy

    def calibrateMethod1(self, calibrationImage, ballPosition):
        """
        Measures the color of the ball and stores it in the class.
        Calibration routine method 1 - Use colors similar to the color of the the pixel at the center of the image to create a mask.
        :param calibrationImage: HSV-image to use for calculation.
        :param ballPosition: Expected ball position in calibrationImage
        :return: None
        """
        verbose = 0

        # The initialization is done with only a small part of the image around the ball position.
        x1 = int(round(ballPosition[0] - calibrationImage.shape[1]/10, 0))
        x2 = int(round(ballPosition[0] + calibrationImage.shape[1]/10, 0))
        y1 = int(round(ballPosition[1] - calibrationImage.shape[0]/10, 0))
        y2 = int(round(ballPosition[1] + calibrationImage.shape[0]/10, 0))
        cropped = calibrationImage[y1:y2, x1:x2]
        # Now the ball is at the center of the cropped image.
        x_center = int(round(cropped.shape[1]/2))
        y_center = int(round(cropped.shape[0]/2))

        # Get the color of the pixel in the image center
        color = calibrationImage[x_center, y_center]

        # Create a mask for the areas with a color similar to the center pixel
        lower_border_arr = color - [20, 20, 20]
        upper_border_arr = color + [20, 20, 20]
        lower_border = tuple(lower_border_arr.tolist())
        upper_border = tuple(upper_border_arr.tolist())
        mask = cv2.inRange(calibrationImage, lower_border, upper_border)

        # Average the color values of the masked area
        colors = cropped[mask == 255]
        h_mean = int(round(np.mean(colors[:, 0])))
        s_mean = int(round(np.mean(colors[:, 1])))
        v_mean = int(round(np.mean(colors[:, 2])))

        avg_color = [h_mean, s_mean, v_mean]

        self.ball_color = avg_color

        if verbose:
            print('Ball HSV color detected: ' + str(self.ball_color))
            cv2.waitKey(0)

    def calibrateMethod2(self, calibrationImage, ballPosition, ratioPxCm, verbose = 0):
        """
        Measures the color of the ball and stores it in the class.
        Calibration routine method 2 - Use HoughCircles to detect a circle in the image and average the colors inside that circle.
        This method also allows to detect and store the radius of the ball, to be used in detectBallPosition method.
        :param calibrationImage: HSV-image to use for calculation.
        :param ballPosition: Expected ball position in calibrationImage
        :param ratioPxCm: Uses the ratio of pixel per cm to filter on the approximate expected ball size
        :return: None
        """

        # The initialization is done with only a small part of the image around the ball position.
        x1 = int(round(ballPosition[0] - calibrationImage.shape[1]/10, 0))
        x2 = int(round(ballPosition[0] + calibrationImage.shape[1]/10, 0))
        y1 = int(round(ballPosition[1] - calibrationImage.shape[0]/10, 0))
        y2 = int(round(ballPosition[1] + calibrationImage.shape[0]/10, 0))
        cropped = calibrationImage[y1:y2, x1:x2]
        # Now the ball is at the center of the cropped image (not used in this method).

        # Some preprocessing
        rgb = cv2.cvtColor(cropped, cv2.COLOR_HSV2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)

        # Detect circles with a size of a ball (radius from 1cm to 2.5cm)
        minRadius = int(1 * ratioPxCm)
        maxRadius = int(5 * ratioPxCm)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, maxRadius, param1=50, param2=30, minRadius=minRadius, maxRadius=maxRadius)

        # Ensure only 1 circle was found
        if circles is not None and circles.shape[0] == 1:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            circle = circles[0]
        else:
            raise Exception('Ball not detected correctly')

        # Create mask with a filled circle
        mask = np.zeros(blurred.shape[:2], dtype="uint8")
        cv2.circle(mask, (circle[0], circle[1]), int(circle[2]), 255, -1)

        # Get average color in mask
        avg_color = cv2.mean(cropped, mask=mask)[:3]

        self.ball_color = avg_color
        self.ball_radius = circle[2]

        if verbose:
            print('Ball HSV color detected: ' + str(self.ball_color))
            print('Ball radius detected (cm): ' + str(self.ball_radius/ratioPxCm))
            # Draw green circle around detected ball
            cv2.circle(rgb, (circle[0], circle[1]), int(circle[2]), (0,255,0), 1)
            cv2.imshow("rgb", rgb)
            cv2.waitKey(0)

    def findMethod0(self, img_hsv, verbose = 0):
        """
        Finds the ball in the image.
        The algorithm is based on the ball color and does not use edge
        recognition to find the ball. As long as the ball color differs from
        the other colors in the image, it works well and is a safe way to find the ball. 
        -- my problem is that the ball color does not differ from other colors in the image --

        First, the image is searched for pixels with similar color to the ball
        color creating a mask. The mask should contain a white point (the ball).
        To ensure that the ball is found, the contours of the mask are found.
        If there are more than one element with contours, a simple
        circle-similarity measure is calculated.
        The element with the highest similarity to a circle is considered as the ball.

        :param img_hsv: HSV-image to find the ball on
        :return: None
        """

        x_mean = []
        y_mean = []
        dist = []
        self.curr_ball_position = (0, 0)

        # Get the areas of the image, which have a similar color to the ball color
        bl_temp=cv2.getTrackbarPos('bl', 'Parameters')
        gl_temp=cv2.getTrackbarPos('gl', 'Parameters')
        rl_temp=cv2.getTrackbarPos('rl', 'Parameters')

        bh_temp=cv2.getTrackbarPos('bh', 'Parameters')
        gh_temp=cv2.getTrackbarPos('gh', 'Parameters')
        rh_temp=cv2.getTrackbarPos('rh', 'Parameters')

        mask=cv2.inRange(img_hsv,(bl_temp,gl_temp,rl_temp),(bh_temp,gh_temp,rh_temp))
        smoothMask = self._smooth_ball_mask(mask)

        # Find contours in the mask, at the moment only one contour is expected
        contours, hierarchy = cv2.findContours(smoothMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # For every contour found, the center is calculated (by averaging the
        # points), and the circle-comparison is done.
        contour_cnt = 0
        for contour in contours:
            contour = contour[:,0,:]
            x_mean.append(int(np.round(np.mean(contour[:,0]))))
            y_mean.append(int(np.round(np.mean(contour[:,1]))))
            contour_cnt += 1

            dist.append(self._check_circle(contour))

        if contour_cnt <= 0 or min(dist) > self.ball_detection_threshold:
            # If there is nothing found or it does not look like a circle, it is
            # assumed that there is no ball in the image.
            self.curr_ball_position = (-1, -1)
            print("No ball detected")  # TODO: give that message to the interface
        else:
            # Otherwise the contour with the best similarity to a circle is chosen
            # to be considered as the ball.
            self.curr_ball_position = (x_mean[np.argmin(dist)], y_mean[np.argmin(dist)])

        self._store_ball_position(self.curr_ball_position)

        if verbose:
            print("Detected ball position: " + str(self.curr_ball_position))
            self._drawMarkerCross(img_hsv, self.curr_ball_position)
            cv2.imshow("img_hsv", img_hsv)
            cv2.imshow("mask", mask)
            cv2.imshow("smoothMask", smoothMask)

    def findMethod1(self, img_hsv):
        """
        Finds the ball in the image.
        The algorithm is based on the ball color and does not use edge
        recognition to find the ball. As long as the ball color differs from
        the other colors in the image, it works well and is a safe way to find the ball. 
        -- my problem is that the ball color does not differ from other colors in the image --

        First, the image is searched for pixels with similar color to the ball
        color creating a mask. The mask should contain a white point (the ball).
        To ensure that the ball is found, the contours of the mask are found.
        If there are more than one element with contours, a simple
        circle-similarity measure is calculated.
        The element with the highest similarity to a circle is considered as the ball.

        :param img_hsv: HSV-image to find the ball on
        :return: None
        """
        verbose = 1

        x_mean = []
        y_mean = []
        dist = []
        self.curr_ball_position = (0, 0)

        # Get the areas of the image, which have a similar color to the ball color
        lower_color = np.asarray(self.ball_color)
        upper_color = np.asarray(self.ball_color)
        lower_color = lower_color - [50, 50, 50]  # good values (for test video are 10,50,50)
        upper_color = upper_color + [50, 255, 255]  # good values (for test video are 10,50,50)
        lower_color[lower_color < 0] = 0
        lower_color[lower_color > 255] = 255
        upper_color[upper_color < 0] = 0
        upper_color[upper_color > 255] = 255

        mask = cv2.inRange(img_hsv, lower_color, upper_color)
        smoothMask = self._smooth_ball_mask(mask)

        # Find contours in the mask, at the moment only one contour is expected
        contours, hierarchy = cv2.findContours(smoothMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # For every contour found, the center is calculated (by averaging the
        # points), and the circle-comparison is done.
        contour_cnt = 0
        for contour in contours:
            contour = contour[:,0,:]
            x_mean.append(int(np.round(np.mean(contour[:,0]))))
            y_mean.append(int(np.round(np.mean(contour[:,1]))))
            contour_cnt += 1

            dist.append(self._check_circle(contour))

        if contour_cnt <= 0 or min(dist) > self.ball_detection_threshold:
            # If there is nothing found or it does not look like a circle, it is
            # assumed that there is no ball in the image.
            self.curr_ball_position = (-1, -1)
            print("No ball detected")  # TODO: give that message to the interface
        else:
            # Otherwise the contour with the best similarity to a circle is chosen
            # to be considered as the ball.
            self.curr_ball_position = (x_mean[np.argmin(dist)], y_mean[np.argmin(dist)])

        self._store_ball_position(self.curr_ball_position)

        if verbose:
            print("Detected ball position: " + str(self.curr_ball_position))
            self._drawMarkerCross(img_hsv, self.curr_ball_position)
            cv2.imshow("img_hsv", img_hsv)

    def findMethod2(self, img_hsv):
        """
        Finds the ball in the image.
        The element with the highest similarity to a circle near ball radius is considered as the ball.

        :param img_hsv: HSV-image to find the ball on
        :return: None
        """
        verbose = 1

        x_mean = []
        y_mean = []
        dist = []
        ballRadiusTolerance = 0.5
        self.curr_ball_position = (0, 0)

        # Get the areas of the image, which have a similar color to the ball color
        lower_color = np.asarray(self.ball_color)
        upper_color = np.asarray(self.ball_color)
        lower_color = lower_color - [50, 50, 50]  # good values (for test video are 10,50,50)
        upper_color = upper_color + [50, 255, 255]  # good values (for test video are 10,50,50)
        lower_color[lower_color < 0] = 0
        lower_color[lower_color > 255] = 255
        upper_color[upper_color < 0] = 0
        upper_color[upper_color > 255] = 255

        rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        mask = cv2.inRange(img_hsv, lower_color, upper_color)
        smoothMask = self._smooth_ball_mask(mask)

        # Find contours in the mask, at the moment only one contour is expected
        contours, hierarchy = cv2.findContours(smoothMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # For every contour found, the center is calculated (by averaging the
        # points), and the circle-comparison is done.
        contour_cnt = 0
        for contour in contours:
            contour = contour[:,0,:]
            x_mean.append(int(np.round(np.mean(contour[:,0]))))
            y_mean.append(int(np.round(np.mean(contour[:,1]))))
            contour_cnt += 1
            # Debug: print contours one by one to find which one is the ball
#            cv2.drawContours(rgb, [contour], 0, (0,255,0), 1)
#            print(contour_cnt)
#            cv2.waitKey(0)
            dist.append(self._check_circle(contour))

            if self._isCircle(contour):
#                print("Found circle")
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                if self.ball_radius * ballRadiusTolerance <= radius <= self.ball_radius * (ballRadiusTolerance + 1):
#                    print("Found ball")
                    cv2.circle(rgb, (int(x), int(y)), int(radius), (0,255,255), 1)
                else:
                    cv2.circle(rgb, (int(x), int(y)), int(radius), (255,0,255), 1)

        if contour_cnt <= 0 or min(dist) > self.ball_detection_threshold:
            # If there is nothin found or it does not look like a circle, it is
            # assumed that there is no ball in the image.
            self.curr_ball_position = (-1, -1)
            print("No ball detected")  # TODO: give that message to the interface
        else:
            # Otherwise the element with the best similarity to a circle is chosen
            # to be considered as the ball.
            self.curr_ball_position = (x_mean[np.argmin(dist)], y_mean[np.argmin(dist)])

        if verbose:
#           print(self.curr_ball_position)
            self._drawMarkerCross(img_hsv, self.curr_ball_position)
#           cv2.imshow("detectBallPosition:mask", img_hsv)
#           print((lower_color,upper_color))
#            cv2.imshow("blurred", blurred)
            cv2.imshow("img_hsv", img_hsv)
            cv2.imshow("mask", mask)
            cv2.imshow("rgb", rgb)
#            cv2.imshow("detectBallPosition:smoothMask", smoothMask)

        self._store_ball_position(self.curr_ball_position)

    def findMethod20(self, field, imgRGB, verbose = 0):
        """
        Finds the ball in the image.
        The element with the highest similarity to a circle near ball radius is considered as the ball.

        :param img_hsv: HSV-image to find the ball on
        :return: None
        """

        x_mean = []
        y_mean = []
        dist = []
        self.curr_ball_position = (0, 0)
        img_hsv = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2HSV)

        # Get the areas of the image, which have a similar color to the ball color
        Hl=cv2.getTrackbarPos('Hl', 'Parameters')
        Sl=cv2.getTrackbarPos('Sl', 'Parameters')
        Vl=cv2.getTrackbarPos('Vl', 'Parameters')

        Hh=cv2.getTrackbarPos('Hh', 'Parameters')
        Sh=cv2.getTrackbarPos('Sh', 'Parameters')
        Vh=cv2.getTrackbarPos('Vh', 'Parameters')

        ballRadiusTolerance = 0.25 * cv2.getTrackbarPos('Radius', 'Parameters')

        mask=cv2.inRange(img_hsv,(Hl,Sl,Vl),(Hh,Sh,Vh))
        smoothMask = self._smooth_ball_mask(mask)

        # Find contours in the mask, at the moment only one contour is expected
        contours, hierarchy = cv2.findContours(smoothMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # For every contour found, the center is calculated (by averaging the
        # points), and the circle-comparison is done.
        contour_cnt = 0
        for contour in contours:
            contour = contour[:,0,:]
            contour_cnt += 1

            if self._isCircle(contour):
                # print("Found circle")
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                if self.ball_radius * ballRadiusTolerance <= radius <= self.ball_radius * (ballRadiusTolerance + 1) and field.inField((int(x), int(y))):
                    # print("Found ball in field")
                    # Criterias : 
                    # - Radius similar to expected - min(radius_expected - radius_observed)
                    # - Closeness to circle shape - min(self._check_circle(contour)) --- not used for now
                    x_mean.append(int(x))
                    y_mean.append(int(y))
                    dist.append(self.ball_radius - radius)

        if contour_cnt == 0 or len(dist) == 0:
            self.curr_ball_position = (-1, -1)
            print("No ball detected")  # TODO: give that message to the interface
        else:
            # Otherwise the element with the best similarity to the ball is chosen
            self.curr_ball_position = (x_mean[np.argmin(dist)], y_mean[np.argmin(dist)])

        if verbose:
            print(str(len(dist)) + " candidate balls detected in image. Coordinates of choosen as the ball: " + str(self.curr_ball_position))
            cv2.imshow("mask", mask)
            cv2.imshow("smoothMask", smoothMask)

        self._store_ball_position(self.curr_ball_position)

    def detectBallPositionMethod3(self, img_hsv):
        """
        Finds the ball in the image.
        The algorithm is based on the ball color and does not use edge
        recognition to find the ball. As long as the ball color differs from
        the other colors in the image, it works well and is a safe way to find the ball. 
        -- my problem is that the ball color does not differ from other colors in the image --

        First, the image is searched for pixels with similar color to the ball
        color creating a mask. The mask should contain a white point (the ball).
        To ensure that the ball is found, the contours of the mask are found.
        If there are more than one element with contours, a simple
        circle-similarity measure is calculated.
        The element with the highest similarity to a circle is considered as the ball.

        :param img_hsv: HSV-image to find the ball on
        :return: None
        """
        verbose = 1

        x_mean = []
        y_mean = []
        dist = []
        ballRadiusTolerance = 0.5
        self.curr_ball_position = (0, 0)

        # Get the areas of the image, which have a similar color to the ball color
        lower_color = np.asarray(self.ball_color)
        upper_color = np.asarray(self.ball_color)
        lower_color = lower_color - [50, 50, 50]  # good values (for test video are 10,50,50)
        upper_color = upper_color + [50, 255, 255]  # good values (for test video are 10,50,50)
        lower_color[lower_color < 0] = 0
        lower_color[lower_color > 255] = 255
        upper_color[upper_color < 0] = 0
        upper_color[upper_color > 255] = 255
#        smoothMask = cv2.erode(mask, None, iterations=2)
#        smoothMask = cv2.dilate(smoothMask, None, iterations=2)
        rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
#        blurred = cv2.GaussianBlur(img_hsv, (5, 5), 1)
        mask = cv2.inRange(img_hsv, lower_color, upper_color)
#        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
#        smoothMask = self._smooth_ball_mask(edges)

        fieldNoBall = cv2.cvtColor(cv2.imread(r'./media/foosballFieldNoBall.png'), cv2.COLOR_BGR2HSV)
        fieldNoBall = cv2.resize(fieldNoBall, (mask.shape[1], mask.shape[0]))
        maskNoBall = cv2.inRange(fieldNoBall, lower_color, upper_color)
        xor = cv2.bitwise_xor(mask, maskNoBall)
        edges = cv2.Canny(xor, 50, 150, apertureSize=3)

        # Find contours in the mask, at the moment only one contour is expected
        contours, hierarchy = cv2.findContours(xor, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # For every contour found, the center is calculated (by averaging the
        # points), and the circle-comparison is done.
        element_ctr = 0
        for element in contours:
            element = element[:,0,:]
            x_mean.append(int(np.round(np.mean(element[:,0]))))
            y_mean.append(int(np.round(np.mean(element[:,1]))))
            element_ctr += 1
            # Debug: print contours one by one to find which one is the ball
#            cv2.drawContours(rgb, [element], 0, (0,255,0), 1)
#            print(element_ctr)
#            cv2.waitKey(0)
            dist.append(self._check_circle(element))

            if self._isCircle(element):
#                print("Found circle")
                ((x, y), radius) = cv2.minEnclosingCircle(element)
                if self.ball_radius * ballRadiusTolerance <= radius <= self.ball_radius * (ballRadiusTolerance + 1):
#                    print("Found ball")
                    cv2.circle(rgb, (int(x), int(y)), int(radius), (0,255,255), 1)
                else:
                    cv2.circle(rgb, (int(x), int(y)), int(radius), (255,0,255), 1)

        if element_ctr <= 0 or min(dist) > self.ball_detection_threshold:
            # If there is nothin found or it does not look like a circle, it is
            # assumed that there is no ball in the image.
            self.curr_ball_position = (-1, -1)
            print("No ball detected")  # TODO: give that message to the interface
        else:
            # Otherwise the element with the best similarity to a circle is chosen
            # to be considered as the ball.
            self.curr_ball_position = (x_mean[np.argmin(dist)], y_mean[np.argmin(dist)])

        if verbose:
#           print(self.curr_ball_position)
#           self._drawMarkerCross(img_hsv, self.curr_ball_position)
#           cv2.imshow("detectBallPosition:mask", img_hsv)
#           print((lower_color,upper_color))
            cv2.imshow("edges", edges)
#            cv2.imshow("blurred", blurred)
#            cv2.imshow("img_hsv", img_hsv)
            cv2.imshow("mask", mask)
            cv2.imshow("maskNoBall", maskNoBall)
            cv2.imshow("xor", xor)
            cv2.imshow("rgb", rgb)
#            cv2.imshow("detectBallPosition:smoothMask", smoothMask)

        self._store_ball_position(self.curr_ball_position)

    def _drawMarkerCross(self, image, position, size=0.05, rgbColor=[0,255,0]):
        """
        :param image: Image on which draw the marker cross
        :param position: Center of the marker cross
        :param size: Size of the cross as a percentage of the image size expressed in decimal (ex: 0,1 for 10%)
        """
        x1 = int(round(position[0] - image.shape[1]*size, 0))
        x2 = int(round(position[0] + image.shape[1]*size, 0))
        y1 = int(round(position[1] - image.shape[0]*size, 0))
        y2 = int(round(position[1] + image.shape[0]*size, 0))

        cv2.line(image, (x1, int(position[1])), (x2, int(position[1])), rgbColor, 2)
        cv2.line(image, (int(position[0]), y1), (int(position[0]), y2), rgbColor, 2)

    def _smooth_ball_mask(self, mask):
        """
        The mask created inDetectBallPosition might be noisy.
        :param mask: The mask to smooth (Image with bit depth 1)
        :return: The smoothed mask
        """
        # create the disk-shaped kernel for the following image processing,
        r = 3
        kernel = np.ones((2*r, 2*r), np.uint8)
        for x in range(0, 2*r):
            for y in range(0, 2*r):
                if(x - r + 0.5)**2 + (y - r + 0.5)**2 > r**2:
                    kernel[x, y] = 0

        # remove noise
        # see http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def _isCircle(self, points):
        """
        :param points: Contour to compare to a circle
        :return: True if contour resembles a circle, false otherwise
        """

        circleMatchingThreshold = 0.1 * cv2.getTrackbarPos('Circle', 'Parameters')

#        if self._check_circle(points) <= self.ball_detection_threshold:
        if self._check_circle(points) <= circleMatchingThreshold:
            return True
        else:
            return False

    def _check_circle(self, points):
        """
        Calculates a comparison value with a circle.
        First, it normalizes the given points, so that their mean is the origin and their distance to the origin
        is 1 in average.
        Then it averages the differences between the points' distance to the origin and 1.
        The resulting value is 0 when the points form a circle, and increases, if there is any deformation.
        It has no upper limit, but will not be smaller than 0.
        To sum up: the lower the value, the better fit the points to a circle
        :param points: the points that mark the contour to check
        :return: Comparison value.
        """
        # Split x- and y-Values into two arrays
        x_vals, y_vals = [], []
        for point in points:
            x_vals.append(point[0])
            y_vals.append(point[1])

        # Shift the circle center to (0,0)
        x_vals = x_vals - np.mean(x_vals)
        y_vals = y_vals - np.mean(y_vals)

        # Bring the circle radius to 1
        radius = np.sqrt((np.sum(x_vals**2 + y_vals**2)) / len(x_vals))
        for point in range(0, len(x_vals)):
            x_vals[point] = x_vals[point]/radius
            y_vals[point] = y_vals[point]/radius

        # Now the result is compared to a unit circle (radius 1), and the
        # differences are averaged.
        dist = np.mean(np.abs(x_vals**2 + y_vals**2 - 1))

        return dist

    ball_position_history = []

    def _store_ball_position(self, ball_position):
        """

        :param ball_position:
        :return:
        """
        if ball_position != (-1, -1):
            self.ball_position_history.append([ball_position])

    def get_var(self, _type):
        """
        Get the class variables
        :param _type: String to choose the variabe
        :return: The requested variable, empty string if requested name is
        unavailable
        """
        if 'ball_position' == _type:
            return self.curr_ball_position
        elif 'ball_position_history' == _type:
            return self.ball_position_history
        else:
            return ""  # False

    def restart(self):
        """
        Clears the ball position history
        :return: None
        """
        self.ball_position_history = []
