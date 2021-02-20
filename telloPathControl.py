import telloControl as telloControl
import cv2

class pathControl:

    LEFT_MOV = "ml 20"
    RIGHT_MOV = "mr 20"
    UP_MOV = "mu 20"
    DOWN_MOV = "md 20"
    # programmed path
    m_path = [
        RIGHT_MOV,
        UP_MOV,
        LEFT_MOV,
        UP_MOV,
        "mf 20",
        RIGHT_MOV,
        UP_MOV,
        LEFT_MOV,
        UP_MOV,
        "mf 20",
        RIGHT_MOV,
        UP_MOV,
        LEFT_MOV,
        UP_MOV,
        "l"
    ]

    m_currentStep = 0
    m_currentPos = [0, 0]
    m_savedPos = [0, 0]
    m_telloControl = None

    m_test = True

    def __init__(self, test, path=None):
        self.m_telloControl = telloControl.telloControl(test)
        self.m_test = test
        if not path is None:
            self.m_path = path

    def pausePath(self):
        self.m_savedPos = self.m_currentPos.copy()

    def resumePath(self):
        self.gotoAbsolute(self.m_savedPos)

    def gotoAbsolute(self, coord):
        hor = coord[0] - self.m_currentPos[0]
        ver = coord[1] - self.m_currentPos[1]
        self.gotoRelative(hor, ver)

    def updateCoordinates(self, hor, ver):
        self.m_currentPos[0] += hor
        self.m_currentPos[1] += ver

        #print("current coordinates ", self.m_currentPos)

    def gotoRelative(self, hor, ver):

        # x axis
        if hor > 0:
            self.m_telloControl.singleCommand("mr " + str(hor))
        elif hor < 0:
            self.m_telloControl.singleCommand("ml " + str(-hor))

        # y axis
        if ver > 0:
            self.m_telloControl.singleCommand("mu " + str(ver))
        elif ver < 0:
            self.m_telloControl.singleCommand("md " + str(-ver))

        # update pos
        self.updateCoordinates(hor, ver)

    def getCurrentCoordinates(self):
        return (self.m_currentPos[0], self.m_currentPos[1])

    def nextPath(self):
        cv2.waitKey(500)
        # get next step
        nextStep = self.m_path[0]
        # remove from programmed path
        self.m_path = self.m_path[1:]

        # send to tello
        command, first_value, stop = self.m_telloControl.singleCommand(nextStep)
        if command == "l":
            cv2.waitKey(0)

        # update coordinates
        if command == "mr":
            self.updateCoordinates(first_value, 0)
        elif  command == "ml":
            self.updateCoordinates(-first_value, 0)
        elif command == "mu":
            self.updateCoordinates(0, first_value)
        elif command == "md":
            self.updateCoordinates(0, -first_value)
        elif command == "cw":
            pass
            # self.updateCoordinates()

        # pending movements
        pending = len(self.m_path)
        finished = pending == 0 or stop
        return finished





