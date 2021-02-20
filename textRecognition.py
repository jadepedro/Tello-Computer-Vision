import cv2
import pytesseract
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils

import utils_mios as utils

class TextRecognition:

    # video FPS
    m_FPS = 25

    # define the two output layer names for the EAST detector model that we are interested
    # the first is the output probabilities and
    # the second can be used to derive the bounding box coordinates of text
    m_layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    # new width and height
    m_new_W = 720
    m_new_H = 720

    # east nn
    m_net = None

    # confidence
    m_confidence = 0.5

    # tick
    m_tick = 0

    # text to search
    m_text = ''

    def __init__(self, text, new_width=320, new_height=320, path='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'):
        # save text
        self.m_text = text

        # save new width and height
        self.m_new_W = new_width
        self.m_new_H = new_height

        # point to Tesseract OCR path
        pytesseract.pytesseract.tesseract_cmd = path

        # load the pre-trained EAST text detector
        print("[INFO] loading EAST text detector...")
        self.m_net = cv2.dnn.readNet("frozen_east_text_detection.pb")
        print("[INFO] loaded EAST")

    def decode_predictions(self, scores, geometry):
        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the
            # geometrical data used to derive potential bounding box
            # coordinates that surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability,
                # ignore it
                if scoresData[x] < self.m_confidence:
                    continue

                # compute the offset factor as our resulting feature
                # maps will be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and
                # then compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height
                # of the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates
                # for the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score
                # to our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # return a tuple of the bounding boxes and associated confidences
        return (rects, confidences)

    def getTextFromVideo(self, img):
        # identify current text
        current_text, cX, cY, size = self.identifyTextfromVideo(img)
        # check if desired text
        if str(self.m_text) == current_text:
            print("current text:", current_text, " on ", cX, " ", cY, " size:", size)
            coord = (cX, cY, size)
            return coord

    def getTextFromJpg(self, img):
        # identify current text
        current_text, cX, cY, size = self.identifyTextfromJpg(img)
        # check if desired text
        if str(self.m_text) == current_text:
            print("current text:", current_text, " on ", cX, " ", cY, " size:", size)
            coord = (cX, cY, size)
            return coord

    def identifyTextfromVideo(self, img):
        text = ""
        centerX, centerY, width, height = 0, 0, 0, 0

        frame = imutils.resize(img, width=720)
        orig = frame.copy()

        (H, W) = frame.shape[:2]
        rW = W / float(self.m_new_W)
        rH = H / float(self.m_new_H)

        #print("h w", H, W)
        #print("newH newW", self.m_new_H, self.m_new_W)

        # resize the frame
        frame = cv2.resize(frame, (self.m_new_W, self.m_new_H))

        # construct a blob from the frame and then perform a forward pass
        # of the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(frame, 1.0, (self.m_new_W, self.m_new_H),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        # set input to nn
        self.m_net.setInput(blob)

        # forward and extract the desired layer results
        (scores, geometry) = self.m_net.forward(self.m_layerNames)

        # decode the predictions, then  apply non-maxima suppression to
        # suppress weak, overlapping bounding boxes
        (rects, confidences) = self.decode_predictions(scores, geometry)
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        max_cropped = None
        max_startX = 0
        max_startY = 0
        max_endX = 0
        max_endY = 0
        max_area = 0
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            area = (endX - startX) * (endY - startY)
            if area > max_area:
                max_startX = startX
                max_startY = startY
                max_endX = endX
                max_endY = endY
                max_area = area

        cropped_img = orig

        if max_area > 3000:
            startX = max_startX
            startY = max_startY
            endX = max_endX
            endY = max_endY
            width = endX - startX
            height = endY - startY
            # crops the image by including a pad proportional to contour size
            padX = int(width * 0.1)
            padY = int(height * 0.1)

            centerX = int(startX + width/2)
            centerY = int(startY + height/2)

            # draw the bounding box on the frame
            cropped_img = orig.copy()
            cropped_img = cropped_img[startY + padY:endY+padY, startX:endX]
            cv2.rectangle(orig, (startX-padX, startY-padY), (endX+padX, endY+padY), (0, 255, 0), 2)

            cv2.circle(orig, (centerX, centerY), 7, (0, 0, 255), -1)
            cv2.circle(orig, (centerX, centerY), 3, (255, 255, 255), -1)

            if cropped_img.size != 0:
                # tune image for improved text recognition
                cropped_img = self.get_grayscale(cropped_img)
                #cropped_img = self.thresholding(cropped_img)q
                #cropped_img = self.apply_brightness_contrast(cropped_img, 32, 32)


                # Apply OCR on the cropped image
                text = pytesseract.image_to_string(cropped_img, config='outputbase digits')
                text = ''.join(e for e in text if e.isalnum())
                cv2.putText(orig, text, (startX + 5, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if cropped_img.size != 0:
            imgStack = utils.stackImages(1, [img, orig, cropped_img])
        else:
            imgStack = utils.stackImages(1, [img, orig])
        cv2.imshow('tello ocr', imgStack)

        self.m_tick += 1
        if self.m_tick == self.m_FPS:
            self.m_tick = 0

        return text, centerX, centerY, width*height

    def identifyTextfromJpg(self, img):
        text="--"

        # Preprocessing the image starts
        # img = cv2.imread("sampleText.jpg")

        # Convert the image to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Performing OTSU threshold
        ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)


        # Specify structure shape and kernel size.
        # Kernel size increases or decreases the area of the rectangle to be detected.
        # A smaller value like (10, 10) will detect each word instead of a sentence.
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

        # Appplying dilation on the threshold image
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

        # Finding contours
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Creating a copy of image
        img_copy = img.copy()

        # Looping through the identified contours
        # Then rectangular part is cropped and passed on to pytesseract for extracting text from it
        # Extracted text is then saved and coordinates of center returned
        max_area, max_x, max_y, max_w, max_h = 0, 0, 0, 0, 0
        centerX, centerY = 0, 0

        # Find biggest text
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w*h
            if area > max_area and area < 300*200:
                centerX = x + w//2
                centerY = y + h//2
                max_x = x
                max_y = y
                max_w = w
                max_h = h
                # Apply OCR on the cropped image
                #cropped = img_copy[y:y + h, x:x + w]
                #text = pytesseract.image_to_string(cropped, config="--psm 10")
                #cv2.putText(img_copy, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if max_w * max_h > 0:
            # Drawing a rectangle on copied image
            rect = cv2.rectangle(img_copy, (max_x, max_y), (max_x + max_w, max_y + max_h), (255, 255, 0), 2)

            # Cropping the text block for giving input to OCR
            cropped = img_copy[max_y:max_y + max_h, max_x:max_x + max_w]

            # Apply OCR on the cropped image
            text = pytesseract.image_to_string(cropped, config="--psm 10")
            text = ''.join(e for e in text if e.isalnum())
            cv2.putText(img_copy, text, (max_x+5, max_y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        #imgStack = utils.stackImages(1, [img, img_copy, cropped])
        imgStack = utils.stackImages(1, [img, img_copy])

        cv2.imshow('tello ocr', imgStack)

        return text, centerX, centerY, max_w*max_y

    def apply_brightness_contrast(self, img, brightness=0, contrast=0):

        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            buf = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)
        else:
            buf = img.copy()

        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

    # preprocessing images

    # get grayscale image
    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def unsharp_mask(self, image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
        """Return a sharpened version of the image, using an unsharp mask."""
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened

    # noise removal
    def remove_noise(self, image):
        return cv2.medianBlur(image, 5)


    # thresholding
    def thresholding(self, image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


    # dilation
    def dilate(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)


    # erosion
    def erode(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)


    # opening - erosion followed by dilation
    def opening(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


    # canny edge detection
    def canny(self, image):
        return cv2.Canny(image, 100, 200)


    # skew correction
    def deskew(self, image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)

        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated


    # template matching
    def match_template(self, image, template):
        return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)













