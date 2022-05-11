import cv2
import numpy as np
from utils_mios import stackImages

class trackBar(object):


    # calibration
    m_H_lower = 20
    m_H_upper = 40
    m_S_lower = 50
    m_S_upper = 255
    m_V_lower = 50
    m_V_upper = 255

    m_lower_color = np.array([58, 150, 183])
    m_upper_color = np.array([60, 255, 255])


    def createTrackBar(self):
        cv2.namedWindow("Calibrate")
        cv2.resizeWindow("Calibrate", 640, 400)

        cv2.createTrackbar("Hue min", "Calibrate", self.m_lower_color[0], 179, self.onTrackBarChange)
        cv2.createTrackbar("Hue max", "Calibrate", self.m_upper_color[0], 179, self.onTrackBarChange)
        cv2.createTrackbar("Sat min", "Calibrate", self.m_lower_color[1], 255, self.onTrackBarChange)
        cv2.createTrackbar("Sat max", "Calibrate", self.m_upper_color[1], 255, self.onTrackBarChange)
        cv2.createTrackbar("Val min", "Calibrate", self.m_lower_color[2], 255, self.onTrackBarChange)
        cv2.createTrackbar("Val max", "Calibrate", self.m_upper_color[2], 255, self.onTrackBarChange)

    def onTrackBarChange(self, value):
        self.m_H_lower = cv2.getTrackbarPos("Hue min", "Calibrate")
        self.m_H_upper = cv2.getTrackbarPos("Hue max", "Calibrate")
        self.m_S_lower = cv2.getTrackbarPos("Sat min", "Calibrate")
        self.m_S_upper = cv2.getTrackbarPos("Sat max", "Calibrate")
        self.m_V_lower = cv2.getTrackbarPos("Val min", "Calibrate")
        self.m_V_upper = cv2.getTrackbarPos("Val max", "Calibrate")

        self.m_lower_color = np.array([self.m_H_lower, self.m_S_lower, self.m_V_lower])
        self.m_upper_color = np.array([self.m_H_upper, self.m_S_upper, self.m_V_upper])

    def maskImage(self, img):
        orig_img = img.copy()
        self.createTrackBar()
        while True:
            # Create image copy

            # Convert to HSV for masking
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # apply mask
            mask = cv2.inRange(imgHSV, self.m_lower_color, self.m_upper_color)
            # Obtain masked image
            masked = cv2.bitwise_and(orig_img, orig_img, mask=mask)
            imgStack = stackImages(1, [[mask, masked]])
            cv2.imshow('mask', imgStack)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
