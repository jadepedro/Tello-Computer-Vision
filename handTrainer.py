import os

import numpy as np
from PIL import Image
import cv2
from djitellopy import Tello


# python -m pip install opencv-contrib-python

def Capture():
    tello = Tello()
    tello.connect()
    tello.streamoff()
    tello.streamon()

    cam = cv2.VideoCapture(0)
    cam.set(3, 320)  # set video width
    cam.set(4, 240)  # set video height

    # For each label, enter id
    hand_id = input('\n enter label end press <return> ==>  ')
    print("\n [INFO] Initializing hand capture. Start taking images and wait ...")
    # Initialize individual sampling face count
    count = 0
    ping = 0
    start = False
    while (True):
        img = telloGetFrame(tello)

        # Save the captured image into the datasets folder
        fileName = "handDroneImages/" + hand_id + "/hand_id." + str(count) + ".jpg"
        if start:
            cv2.imwrite(fileName, img)

        cv2.putText(img, str(count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
        cv2.putText(img, fileName, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        cv2.imshow('image', img)

        k = cv2.waitKey(500) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
        elif k == ord('s'):
            start = True

        if start:
            count += 1
        # keep-alive drone
        ping +=1
        if ping == 20:
            tello.streamon()
            ping = 0

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

# function to get the images and label data

def telloGetFrame(myDrone, w=320, h=240):
    myFrame = myDrone.get_frame_read()
    myf = myFrame.frame
    img = cv2.resize(myf, (w, h))
    return img

Capture()