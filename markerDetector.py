import cv2
from trackBar import trackBar
import numpy as np

from faceDetectionDrawUtilities import GREEN_COLOR

class markerDetector(object):


    # tracker bar for debugging
    m_tracker = None

    m_lower_mask = np.array([58, 150, 183])
    m_upper_mask = np.array([60, 255, 255])

    def __init__(self):
        pass

    def detectMarkers(self, img):
        #self.m_tracker = trackBar()

        # Convert to HSV for masking
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #greenBGR = np.uint8([[[0, 255, 0]]])
        #greenHSV = cv2.cvtColor(greenBGR, cv2.COLOR_BGR2HSV)
        # Build green mask
        mask = cv2.inRange(imgHSV, self.m_lower_mask, self.m_upper_mask)
        # Obtain masked image
        masked = cv2.bitwise_and(img, img, mask=mask)
        # show resized
        h, w, c = masked.shape
        ratio = h/w
        # cv2.namedWindow('marker detection', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        masked_res = cv2.resize(masked, (1920, int(1920 * ratio)))
        cv2.imshow('marker detection', masked_res)

        # find contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # the number of contours are the number of recognized faces
        return len(contours)





