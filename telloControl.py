import keyboard

from djitellopy import Tello

from time import sleep

class telloControl(object):

    # -- command control --
    m_commandDict = {
        "ml": ("moves drone to left (cm)", "moving left"),
        "mr": ("moves drone to right (cm)", "moving right"),
        "mb": ("moves drone back (cm)", "moving back"),
        "mf": ("moves drone front (cm)", "moving front"),
        "mu": ("moves drone up (cm)", "moving up"),
        "md": ("moves drone down (cm)", "moving down"),
        "t": ("take off drone", "taking off"),
        "l": ("land drone", "landing"),
        "f": ("flip drone (direction)", "flipping drone"),
        "ccw": ("rotate drone ccw (degrees)", "rotating ccw"),
        "cw": ("rotate drone cw (degrees)", "rotating cw"),
        "exit": ("exit drone control", "exiting")
    }

    # -- keyboard control --
    # key help
    m_keyDict = {
        "a": ("moves drone to left", "moving left"),
        "d": ("moves drone to right", "moving right"),
        "s": ("moves drone back", "moving back"),
        "x": ("moves drone front", "moving front"),
        "k": ("moves drone up", "moving up"),
        "m": ("moves drone down", "moving down"),
        "q": ("take off drone", "taking off"),
        "p": ("land drone", "landing"),
        "j": ("rotate drone ccw 45", "rotating ccw"),
        "l": ("rotate drone cw 45", "rotating cw"),
        "f": ("flip using opossite to last movement", "flipping drone"),
        "2-5": ("set speed", "setting speed"),
        "y": ("get current speed", "getting speed")
    }
    # number of centimeters to move each time a key is pressed
    m_step = 30
    # rotating degrees
    m_rotatestep = 45
    # last direction of movement
    m_flip = 'f'

    # -- PID control --


    # -- general parameters --
    # tello object
    m_tello = None
    # test mode
    m_test = False


    def __init__(self, test, tello = None):
        if tello is None:
            self.m_tello = Tello()
        self.m_test = test
        if not self.m_test:
            self.m_tello.connect()
            #self.m_tello.set_speed(0)


    # Use keys to control drone
    def keyControl(self):
        self.__printKeyHelp()
        while True:
            try:  # used try so that if user pressed other than the given key error will not be shown
                key = keyboard.read_key()

                if key in self.m_keyDict:
                    print(key, ' - ', self.m_keyDict[key][1])

                # Update internal values - not to be sent to tello
                if key in ('2', '3', '4', '5'):
                    self.setKeySpeed(key)
                elif key in ('y'):
                    self.getKeySpeed()
                # tello function - to be sent to tello
                elif key in self.m_keyDict and not self.m_test:
                    if key == 'a':
                        self.m_tello.move_left(self.m_step)
                        self.m_flip = 'r'
                    elif key == 'd':
                        self.m_tello.move_right(self.m_step)
                        self.m_flip = 'l'
                    elif key == 's':
                        self.m_tello.move_back(self.m_step)
                        self.m_flip = 'f'
                    elif key == 'x':
                        self.m_tello.move_forward(self.m_step)
                        self.m_flip = 'b'
                    elif key == 'j':
                        self.m_tello.rotate_counter_clockwise(self.m_rotatestep)
                    elif key == 'l':
                        self.m_tello.rotate_clockwise(self.m_rotatestep)
                    elif key == 'k':
                        self.m_tello.move_up(self.m_step)
                    elif key == 'm':
                        self.m_tello.move_down(self.m_step)
                    elif key == 'q':
                        self.m_tello.takeoff()
                    elif key == 'p':
                        self.m_tello.land()
                    elif key == 'f':
                        self.m_tello.flip(self.m_flip)
                elif key == 'esc':
                    # land as a security measure
                    self.m_tello.land()
                    break  # finishing the loop
                sleep(0.3) #avoid repeated key
            except ValueError:
                print(ValueError)
                # land as a security measure
                self.m_tello.land()
                break  # if user pressed a key other than the given key the loop will break

    def singleCommand(self, commands):

        stop = False
        first_value = 0

        command_split = commands.split(' ')
        command = command_split[0]
        print("command: ", command)
        args = ''
        if len(command_split) > 1:
            args = command_split[1:]
            first_value = args[0]
            # second_value = int(args[1])
            #print(first_value)

        if command in self.m_commandDict:
            print(command, ' - ', self.m_commandDict[command][1], ' ', args)

        # tello function - to be sent to tello
        if command in self.m_commandDict and not self.m_test:
            if command == 'ml':
                self.m_tello.move_left(int(first_value))
            elif command == 'mr':
                self.m_tello.move_right(int(first_value))
            elif command == 'mb':
                self.m_tello.move_back(int(first_value))
            elif command == 'mf':
                self.m_tello.move_forward(int(first_value))
            elif command == 'ccw':
                self.m_tello.rotate_counter_clockwise(int(first_value))
            elif command == 'cw':
                self.m_tello.rotate_clockwise(int(first_value))
            elif command == 'mu':
                self.m_tello.move_up(int(first_value))
            elif command == 'md':
                self.m_tello.move_down(int(first_value))
            elif command == 't':
                self.m_tello.takeoff()
            elif command == 'l':
                self.m_tello.land()
            elif command == 'f':
                self.m_tello.flip(first_value)
            else:
                print("command not recognized")
        elif command == 'exit':
            # land as a security measure
            if not self.m_test:
                self.m_tello.land()
            stop = True  # finishing the loop

        return command, int(first_value), stop

    def commandControl(self):
        self.__printCommandHelp()
        stop = False

        while True:
            try:
                commands = input()

                # Update internal values - not to be sent to tello
                _, _ , stop = self.singleCommand(commands)

            except ValueError:
                print(ValueError)
                # land as a security measure
                self.m_tello.land()
                break  # if user pressed a key other than the given key the loop will break

    def __printKeyHelp(self):
        """
        Prints the help based on the key dictionary
        :return:
        """
        print("\n-----------")
        print("Keys: ")
        for entry in self.m_keyDict:
            print(entry, " - ", self.m_keyDict[entry][0])
        print("-----------\n\n")

    def __printCommandHelp(self):
        """
        Prints the help based on the command dictionary
        :return:
        """
        print("\n-----------")
        print("Commands: ")
        for entry in self.m_commandDict:
            print(entry, " - ", self.m_commandDict[entry][0])
        print("-----------\n\n")

    def getKeySpeed(self):
        print("current speed", self.m_step)
        return self.m_step

    def setKeySpeed(self, key):
        ret =int(key)*10
        print("set speed to", ret)
        return ret
