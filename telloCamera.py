import numpy as np
import cv2
import json
import tensorflow as tf
from PIL import Image

from djitellopy import Tello
from telloPIDControl import telloPIDControl
from textRecognition import TextRecognition
import objectTagger
import telloPathControl as telloPathControl
from tensorflowmodel.HandNumbersTensorFlow.tf_prediction import Model


import utils_mios as utils

class telloCamera:
    # test mode
    m_test = False
    m_fly = True
    m_tookoff = False
    m_useDroneCamera = False

    # Haar cascade
    m_cascade = None

    # laptop camera for test mode
    m_capLaptop = None

    # color to search (HSV tuple)
    m_color = 30
    # HUE threshold to both sides of m_color
    m_threshold = 10
    # S and H upper and lower limits

    m_H_lower = 20
    m_H_upper = 40
    m_S_lower = 50
    m_S_upper = 255
    m_V_lower = 50
    m_V_upper = 255
    # upper and lower bounds
    m_lower_color = np.array([37, 80, 0])
    m_upper_color = np.array([70, 255, 255])

    # Size for viewing
    m_camera_width = 320
    m_camera_height = 240

    m_frameContour = None

    m_tello = None

    # PID controller
    m_telloPIDControl = None

    # Current target color bounds
    m_lower_targetColor = np.array([0, 80, 0])
    m_upper_targetColor = np.array([14, 255, 255])

    # average of box size to smooth reported size
    m_average_box = np.zeros(10)

    # target function for PID control
    m_target_function = None

    # face recognizer
    m_recognizer = None
    # names related to ids: example ==> Marcelo: id=1,  etc
    m_names = ['None', 'Javier', 'Diego', 'Luna', 'Mari']

    def __init__(self, test, trackfunction="Face", useDroneCamera = True):
        print("use Drone Camera: ", useDroneCamera)
        self.m_useDroneCamera = useDroneCamera
        self.m_tello = Tello()
        self.m_test = test
        self.m_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default_b.xml")
        self.m_telloPIDControl = telloPIDControl(self.m_tello, (320, 240), self.m_test)
        self.m_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.m_recognizer.read('trainer/trainer.yml')
        if not self.m_test:
            self.m_tello.connect()
        else:
            self.m_capLaptop = cv2.VideoCapture(0)

        if trackfunction == "Face":
            self.m_target_function = self.getFace
        elif trackfunction == "Object":
            self.m_target_function = self.getObject

    def telloGetFrame(self, myDrone, w=320, h=240):
        img = None
        try:
            if self.m_useDroneCamera:
                myFrame = myDrone.get_frame_read()
                myf = myFrame.frame
                img = cv2.resize(myf, (w, h))
            else:
                _, myf = self.m_capLaptop.read()
                img = cv2.resize(myf, (w, h))
        except ValueError:
            pass
        return img

    def initializeTello(self):
        #TODO: Remove -> included in the __init__
        myDrone = None
        if self.m_useDroneCamera:
            # CONNECT TO TELLO
            myDrone = Tello()
            myDrone.connect()
            myDrone.streamoff()
            myDrone.streamon()
        return myDrone

    def onTrackBarChange(self, value):
        self.m_H_lower = cv2.getTrackbarPos("Hue min", "Calibrate")
        self.m_H_upper = cv2.getTrackbarPos("Hue max", "Calibrate")
        self.m_S_lower = cv2.getTrackbarPos("Sat min", "Calibrate")
        self.m_S_upper = cv2.getTrackbarPos("Sat max", "Calibrate")
        self.m_V_lower = cv2.getTrackbarPos("Val min", "Calibrate")
        self.m_V_upper = cv2.getTrackbarPos("Val max", "Calibrate")
        self.m_lower_color = np.array([self.m_H_lower, self.m_S_lower, self.m_V_lower])
        self.m_upper_color = np.array([self.m_H_upper, self.m_S_upper, self.m_V_upper])

    def calibrate(self):
        """
        tool to calibrate color
        """
        cv2.namedWindow("Calibrate")
        cv2.resizeWindow("Calibrate", 640, 300)
        cv2.createTrackbar("Hue min", "Calibrate", self.m_lower_targetColor[0], 179, self.onTrackBarChange)
        cv2.createTrackbar("Hue max", "Calibrate", self.m_upper_targetColor[0], 179, self.onTrackBarChange)
        cv2.createTrackbar("Sat min", "Calibrate", self.m_lower_targetColor[1], 255, self.onTrackBarChange)
        cv2.createTrackbar("Sat max", "Calibrate", self.m_upper_targetColor[1], 255, self.onTrackBarChange)
        cv2.createTrackbar("Val min", "Calibrate", self.m_lower_targetColor[2], 255, self.onTrackBarChange)
        cv2.createTrackbar("Val max", "Calibrate", self.m_upper_targetColor[2], 255, self.onTrackBarChange)
        self.startVideoLoopTarget()

    def startVideoLoopTarget(self, tookoff_=False, fly_=False, initialize_=True):
        try:
            tookoff = tookoff_
            self.m_fly = fly_
            if initialize_:
                self.m_tello = self.initializeTello()
            m_tick = 0

            while True:
                # Open VideoStream
                frame = self.telloGetFrame(self.m_tello, 320, 240)

                # Call configured targeting function
                (cX, cY, boxArea) = self.m_target_function(frame)

                if m_tick == 5:  # to avoid flooding with too many commands
                    m_tick = 0
                    self.m_telloPIDControl.trackTargetBoxed((cX, cY), boxArea)
                else:
                    m_tick += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    if self.m_fly:
                        self.m_tello.land()
                    break

                if cv2.waitKey(1) & 0xFF == ord('f'):
                    print("taking off")
                    if not self.m_test and not tookoff:
                        self.m_fly = True
                        self.m_telloPIDControl.setFly(self.m_fly)
                        self.m_tello.takeoff()
                        self.m_tello.move_up(80)
                        tookoff = True

            if self.m_test:
                self.m_capLaptop.release()

        except ValueError:
            if not self.m_tello is None:
                self.m_tello.land()
        cv2.destroyAllWindows()

    def putName(self, img, img_grey, x, y, w, h):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = self.m_recognizer.predict(img_grey[y:y + h, x:x + w])

        # If confidence is less them 100 ==> "0" : perfect match
        if confidence < 100:
            id = self.m_names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

    def getFace(self, img):
        # review https://towardsdatascience.com/real-time-face-recognition-an-end-to-end-project-b738bb0f7348
        img_copy = img.copy()

        # convert to grey
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        coord = np.array([0, 0, 0])
        faces_coord = self.m_cascade.detectMultiScale(img_grey, scaleFactor=1.3, minNeighbors=5)

        # loop through faces to find the one with greatest area
        max_area = 0
        for (x, y, width, height) in faces_coord:

            self.putName(img_copy, img_grey, x, y, width, height)

            area = round(width * height, -3)
            if area > max_area:
                # update average
                self.m_average_box[:9] = self.m_average_box[1:]
                self.m_average_box[9] = area
                # print("area:", self.m_average_box)
                max_area = area
                # return coordinates, including box area
                coord = (int(x + (width/2)), int(y + (height/2)), np.average(self.m_average_box))
            # draw rectangle
            cv2.rectangle(img_copy, (x, y), (x + width, y + height), (0, 255, 0), 3)

        # show image processing
        imgStack = utils.stackImages(1, [[img, img_copy]])
        cv2.imshow('tello', imgStack)

        return coord

    def getObject(self, img):
        frameContour = img.copy()
        # convert to HSV
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # create mask
        mask = cv2.inRange(imgHSV, self.m_lower_color, self.m_upper_color)
        # apply mask
        masked = cv2.bitwise_and(img, img, mask=mask)

        # find contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop through contours to find the one with highest area
        max_area = 300
        max_cnt = None
        coord = np.array([0, 0, 0])
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                self.m_average_box[:9] = self.m_average_box[1:]
                self.m_average_box[9] = area
                max_area = area
                max_cnt = cnt
        if not max_cnt is None:
                M = cv2.moments(max_cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                coord = (cX, cY, np.average(self.m_average_box))

                # draw the center of the shape on the image
                cv2.circle(frameContour, (cX, cY), 7, (0, 0, 255), -1)
                cv2.circle(frameContour, (cX, cY), 3, (255, 255, 255), -1)

                cv2.drawContours(frameContour, max_cnt, -1, (0, 255, 0), 3)

        imgStack = utils.stackImages(1, [[img, frameContour], [mask, masked]])
        cv2.imshow('tello calibration', imgStack)

        return coord

    def startVideoLoopSearchShapes(self, usemask = True):
        taggerB = objectTagger.objectTagger(False)
        taggerG = objectTagger.objectTagger(False)
        taggerR = objectTagger.objectTagger(False)
        #taggerG.createTrackBar()
        pathControl = telloPathControl.pathControl(self.m_test)
        finished = False

        # order and position
        order = {1: "blue", 2: "green", 3: "red"}
        coordinates = {
            "green": [-1, -1],
            "red": [-1, -1],
            "blue": [-1, -1]
        }

        try:
            tookoff = False
            self.m_fly = False
            self.m_tello = self.initializeTello()
            m_tick = 0
            cXB, cYB, cXG, cYG, cXR, cYR = 0, 0, 0, 0, 0, 0
            tagB, tagG, tagR = "", "", ""

            # index of the object to be searched
            shape_index = 1
            red_found = False
            green_found = False
            blue_found = False

            paused = False
            while True:

                if shape_index <= len(order):
                    current_shape = order[shape_index]
                    #print ("---- now searching: ", current_shape)
                    if current_shape == "red":
                        taggerR.setNumber(shape_index)
                        taggerB.setNumber(0)
                        taggerG.setNumber(0)
                    elif current_shape == "green":
                        taggerR.setNumber(0)
                        taggerB.setNumber(0)
                        taggerG.setNumber(shape_index)
                    elif current_shape == "blue":
                        taggerR.setNumber(0)
                        taggerB.setNumber(shape_index)
                        taggerG.setNumber(0)

                # if there is a registered coordinate, go directly and pause search path
                if coordinates[current_shape][0] > -1:
                    print("... pausing path...")
                    pathControl.pausePath()
                    paused = True
                    pathControl.gotoAbsolute(coordinates[current_shape])


                # Open VideoStream
                frame = self.telloGetFrame(self.m_tello, 320, 240)

                if True: #not red_found:
                    cXR, cYR, tagR = taggerR.tagColor(frame, ("red", "triangle"))
                    if (cXR != 0 or cYR != 0) and not red_found:
                        print("found red on ", cXR, ", ", cYR, " coordinates ", pathControl.getCurrentCoordinates())
                        coordinates["red"] = pathControl.getCurrentCoordinates()
                        if current_shape == "red":
                            red_found = True
                            print(str(shape_index) + "--->" + "red")
                            shape_index += 1

                if True: #not blue_found:
                    cXB, cYB, tagB = taggerB.tagColor(frame, ("blue", "circle"))
                    if (cXB != 0 or cYB != 0):
                        print("found blue on ", cXB, ", ", cYB, " coordinates ", pathControl.getCurrentCoordinates())
                        coordinates["blue"] = pathControl.getCurrentCoordinates()
                        if current_shape == "blue":
                            blue_found = True
                            print(str(shape_index) + "--->" + "blue")
                            shape_index += 1

                if True: #not green_found:
                    cXG, cYG, tagG = taggerG.tagColor(frame, ("green", "rectangle"))
                    if (cXG != 0 or cYG != 0):
                        print("found green on ", cXG, ", ", cYB, " coordinates ", pathControl.getCurrentCoordinates())
                        coordinates["green"] = pathControl.getCurrentCoordinates()
                        if current_shape == "green":
                            green_found = True
                            print(str(shape_index) + "--->" + "green")
                            shape_index += 1
                if shape_index > 3:
                    finished = True
                    cv2.waitKey(0)
                    if self.m_useDroneCamera:
                        self.m_tello.land()
                if paused and not finished:
                    print("... resuming path...")
                    pathControl.resumePath()

                if tookoff:
                    finished = pathControl.nextPath()
                if finished:
                    cv2.waitKey(0)
                    #pathControl.gotoAbsolute((0,0))


                if cv2.waitKey(2) & 0xFF == ord('q') or finished:
                    if not self.m_test:
                        self.m_tello.land()
                        cv2.waitKey(0)
                    break
                if cv2.waitKey(2) & 0xFF == ord('f'):
                    print("taking off")
                    if not self.m_test and not tookoff:
                        self.m_fly = True
                        self.m_tello.takeoff()
                        tookoff = True

            if self.m_test:
                self.m_capLaptop.release()
        except ValueError:
            self.m_tello.land()
        cv2.waitKey(0)

    def startVideoLoopSearchHand(self):
        # Load model
        print("loading model")
        model = Model()
        print("loaded")

        path = [
            "ccw 180",

            "mu 20",
            "ml 25",
            "mu 25",
            "mr 25",

            "mu 20",
            "ml 25",
            "mu 20",
            "mr 25",

            "mu 20",
            "ml 25",
            "mu 20",
            "mr 25",

            "l"
        ]

        try:
            self.m_tookoff = False
            self.m_fly = False
            self.m_tello = self.initializeTello()

            # 1. Identify hand
            while True:
                # Open VideoStream
                frame = self.telloGetFrame(self.m_tello, 320, 240)

                # identify number
                num, _ = self.identifyNumber(frame, model)

                # Common keyboard handling
                if self.handleKeyboard():
                    break

            number = str(num)

            # after the user has indicated she´s happy with the number recgonized...

            # 2. Search text through search path
            found = False
            finished = False
            textRecognition = TextRecognition(number)
            pathControl = telloPathControl.pathControl(self.m_test, path)

            print("searching: ", number)

            while True:
                # Open VideoStream
                frame = self.telloGetFrame(self.m_tello, 320, 240)
                # Recognize text
                text, centerX, centerY, area = textRecognition.identifyTextfromJpg(frame)
                print("text on image:>", text, "<")

                # Common keyboard handling
                if self.handleKeyboard():
                    break

                # Compare text with requested. if matches, stop and initiate PID
                if text == str(number):
                    print("found text:>", text,"<")
                    found = True
                    # center image: stop and switch to PID control
                    break
                #if not, keep searching
                else:
                    finished = pathControl.nextPath()
                    if finished:
                        break
            #TODO As experiment 3. When found, use PID

            if found:
                # TODO As experiment 3. When found, use PID
                # set text as the target function
                #self.m_target_function = textRecognition.getTextFromJpg
                #self.startVideoLoopTarget(self.m_tookoff, self.m_fly, initialize_= False)
                frame = self.telloGetFrame(self.m_tello, 320, 240)
                cv2.putText(frame, "Found text:" + number, (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if self.m_fly and self.m_useDroneCamera:
                    self.m_tello.land()

                cv2.imshow("Found", frame)
            else:
                print("Failed to found target ", number)

            cv2.waitKey(0)

        except ValueError:
            if not self.m_tello is None:
                self.m_tello.land()
        cv2.destroyAllWindows()

    def identifyNumber(self, frame, model):
        # convert to PIL format
        frame = cv2.flip(frame, 1)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_copy = frame.copy()

        im_pil = Image.fromarray(img)
        # Identify hand https://stackoverflow.com/questions/37785641/how-to-find-max-value-in-array-of-dictionary
        outputs = model.predict(im_pil)
        from operator import itemgetter
        max_predicted_item = max(outputs['predictions'], key=itemgetter('confidence'))
        prediction = max_predicted_item['label']
        confidence = max_predicted_item['confidence']

        # output
        if int(confidence) > 0.90:
            cv2.putText(img_copy, prediction, (160, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(img_copy, str(confidence), (160, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
        imgStack = utils.stackImages(1, [frame, img_copy])
        cv2.imshow('tello hand', imgStack)

        # print("prediction:", prediction, " confidence:", confidence)

        return prediction, confidence

    def handleKeyboard(self):
        breakme = False
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if self.m_fly:
                self.m_tello.land()
            breakme = True

        elif key == ord('f'):
            print("taking off")
            if not self.m_test and not self.m_tookoff:
                self.m_fly = True
                self.m_telloPIDControl.setFly(self.m_fly)
                self.m_tello.takeoff()
                #self.m_tello.move_up(50)
                self.m_tookoff = True

        elif key == ord('n'):
            print ("next step")
            breakme = True

        return breakme













