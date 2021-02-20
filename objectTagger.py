import cv2
import numpy as np
import utils_mios as utils

class objectTagger:

    m_test = False

    # hue tags https://en.wikipedia.org/wiki/Hue
    m_colorHueTags = {
        0: ("red", 30),
        7: ("orange red", 45),
        15: ("orange", 59),

    }

    # calibration
    m_H_lower = 20
    m_H_upper = 40
    m_S_lower = 50
    m_S_upper = 255
    m_V_lower = 50
    m_V_upper = 255
    # upper and lower bounds
    m_lower_color = np.array([37, 80, 0])
    m_upper_color = np.array([70, 255, 255])


    red1_lower_color = np.array([0, 207, 0])
    red1_upper_color = np.array([20, 255, 255])
    red2_lower_color = np.array([131, 171, 0])
    red2_upper_color = np.array([179, 255, 255])
    blue_lower_color = np.array([87, 171, 0])
    blue_upper_color = np.array([125, 255, 255])
    green_lower_color = np.array([36, 76, 23])
    green_upper_color = np.array([122, 255, 255])

    m_shapeTag = {
        3: "triangle",
        4: "rectangle",
        5: "pentagon",
        6: "hexagon",
        999: "circle",
        -1: "unknow"
    }

    m_number = 0

    def __init__(self, test):
        self.m_test = test

    def tagObjects(self, img):
        pass

    def setNumber(self, number):
        self.m_number = number

    def tagShape(self, img, mask, tagcolor="", tagshape=""):
        # convert to gray and blur
        if not mask is None:
            imgMasked = cv2.bitwise_and(img, img, mask = mask)
        # imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # imgBlur = cv2.GaussianBlur(imgMasked, (7, 7), 1)

        # contours
        #imgCanny = cv2.Canny(img, 50, 50)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print("num contours:", len(contours))
        imgContour = img.copy()

        # find bigger contour
        max_area = 0
        max_cnt = None
        max_p1 = None
        max_p2 = None
        max_cX = 0
        max_cY = 0
        max_tag = ""
        max_approx = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                #cv2.drawContours(imgContour, cnt, -1, (0, 255, 0), 3)
                tag, boundingrect, approx = self._identifyTag(cnt, area)

                if tagshape == tag or tagshape=="":
                    p1 = (boundingrect[0], boundingrect[1])
                    p2 = (boundingrect[0] + boundingrect[2], boundingrect[1] + boundingrect[3])
                    cv2.rectangle(imgContour, p1, p2, (255, 0, 0), 2)
                    if self.m_number == 0:
                        cv2.putText(imgContour, tagcolor + " " + tag, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 50, 0), 2)
                    else:
                        cv2.putText(imgContour, str(self.m_number), (160, 120,), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 0), 3)
                        cv2.putText(imgContour, str(self.m_number) + " " + tagcolor + " " + tag, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 50, 0), 2)
                    cv2.imshow(tagcolor + " " + tagshape, imgContour)

                    if area > max_area:
                        max_area = area
                        max_cnt = cnt
                        max_p1 = p1
                        max_p2 = p2
                        max_cX = int(boundingrect[0] + boundingrect[2]/2)
                        max_cY = int(boundingrect[1] + boundingrect[3]/2)
                        max_tag = tag
                        max_approx = approx

        # identify the one with larger area
        cv2.drawContours(imgContour, max_cnt, -1, (0, 0, 255), 5)
        cv2.drawContours(imgContour, max_approx, -1, (255, 255, 255), 20)
        cv2.rectangle(imgContour, max_p1, max_p2, (0, 0, 255), 2)
        # draw the center of the shape on the image
        cv2.circle(imgContour, (max_cX, max_cY), 7, (0, 0, 255), -1)
        cv2.circle(imgContour, (max_cX, max_cY), 3, (255, 255, 255), -1)

        imgStack = utils.stackImages(0.9, [[img, imgContour], [mask, imgMasked]])
        #imgStack = utils.stackImages(0.9, [[img, imgMasked, imgContour], [mask, imgCanny, imgGray]])
        cv2.imshow("shapes"+tagcolor, imgStack)

        # print (max_cX, " ", max_cY, " ", max_tag)
        return max_cX, max_cY, max_tag

    def _identifyTag(self, cnt, area):
        perimeter = cv2.arcLength(cnt, True)

        # approximate to poligon
        approx = cv2.approxPolyDP(cnt, 0.02*perimeter, True)
        num_points = len(approx)

        # identify a bounding rectangle
        boundingrect = cv2.boundingRect(approx)

        # circle-ness (without using Hough detection)
        radius = perimeter / (2 * np.pi)
        circle_area = np.pi * radius**2
        isCircle = circle_area / area < 1.6

        # identify shape
        if num_points >= 3 and num_points <=6:
            ret = (self.m_shapeTag[num_points])
        elif isCircle:
            ret = (self.m_shapeTag[999])
        else:
            ret = (self.m_shapeTag[-1])

        #print(ret, "circle area/area ratio:", circle_area / area, " num points: ", num_points)

        return ret, boundingrect, approx

    def tagColor(self, img, targetTags):

        color = targetTags[0]
        shape = targetTags[1]

        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cX, cY = 0, 0
        tag = ""

        if color == "red":
            mask_red1 = cv2.inRange(imgHSV, self.red1_lower_color, self.red1_upper_color)
            mask_red2 = cv2.inRange(imgHSV, self.red2_lower_color, self.red2_upper_color)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            #masked_red = cv2.bitwise_and(imgHSV, imgHSV, mask=mask_red)
            cX, cY, tag = self.tagShape(img, mask_red, color, shape)
        elif color == "blue":
            mask_blue = cv2.inRange(imgHSV, self.blue_lower_color, self.blue_upper_color)
            #masked_blue = cv2.bitwise_and(imgHSV, imgHSV, mask=mask_blue)
            cX, cY, tag = self.tagShape(img, mask_blue, color, shape)
        elif color == "green":
            mask_green = cv2.inRange(imgHSV, self.green_lower_color, self.green_upper_color)
            #masked_green = cv2.bitwise_and(imgHSV, imgHSV, mask=mask_green)
            cX, cY, tag = self.tagShape(img, mask_green, color, shape)
        elif color == "":
            mask_red = cv2.inRange(imgHSV, self.red_lower_color, self.red2_upper_color)
            mask_blue = cv2.inRange(imgHSV, self.blue_lower_color, self.blue_upper_color)
            mask_green = cv2.inRange(imgHSV, self.green_lower_color, self.green_upper_color)
            mask = cv2.bitwise_or(mask_red, mask_blue)
            mask = cv2.bitwise_or(mask, mask_green)
            cX, cY, tag = self.tagShape(img, mask, color, shape)
        #print (cX, " ", cY, " ", tag)
        return cX, cY, tag

    def onTrackBarChange(self, value):
        self.m_H_lower = cv2.getTrackbarPos("Hue min", "Calibrate")
        self.m_H_upper = cv2.getTrackbarPos("Hue max", "Calibrate")
        self.m_S_lower = cv2.getTrackbarPos("Sat min", "Calibrate")
        self.m_S_upper = cv2.getTrackbarPos("Sat max", "Calibrate")
        self.m_V_lower = cv2.getTrackbarPos("Val min", "Calibrate")
        self.m_V_upper = cv2.getTrackbarPos("Val max", "Calibrate")

        self.m_lower_color = np.array([self.m_H_lower, self.m_S_lower, self.m_V_lower])
        self.m_upper_color = np.array([self.m_H_upper, self.m_S_upper, self.m_V_upper])

        self.green_lower_color = self.m_lower_color
        self.green_upper_color = self.m_upper_color

    def createTrackBar(self):
        cv2.namedWindow("Calibrate")
        cv2.resizeWindow("Calibrate", 640, 400)

        self.m_lower_color = self.green_lower_color
        self.m_upper_color = self.green_upper_color

        cv2.createTrackbar("Hue min", "Calibrate", self.m_lower_color[0], 179, self.onTrackBarChange)
        cv2.createTrackbar("Hue max", "Calibrate", self.m_upper_color[0], 179, self.onTrackBarChange)
        cv2.createTrackbar("Sat min", "Calibrate", self.m_lower_color[1], 255, self.onTrackBarChange)
        cv2.createTrackbar("Sat max", "Calibrate", self.m_upper_color[1], 255, self.onTrackBarChange)
        cv2.createTrackbar("Val min", "Calibrate", self.m_lower_color[2], 255, self.onTrackBarChange)
        cv2.createTrackbar("Val max", "Calibrate", self.m_upper_color[2], 255, self.onTrackBarChange)