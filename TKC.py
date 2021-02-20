import argparse
import cv2

from telloControl import telloControl
from telloCamera import telloCamera

# Gestion de argumentos
def buildArgParser():
    # Instancia para los argumentos
    parser = argparse.ArgumentParser()
    # Añade los argumentos permitidos
    parser.add_argument("-m", default='k', nargs='?', choices=['k', 'c', 's', 'obj', 'p', 'face', 'r'],
                        help="Enable keyboard control (k) / command control (c) / search color (s)\n \
                        /object track (obj) /path search (p) /face track (face)\n \
                        /hand number recognition (r)")
    parser.add_argument("-t", action='store_true', help="Enable test mode: no command sent")
    parser.add_argument("-l", action='store_true', help="Enable laptop camera")
    args = parser.parse_args()

    return args.m, args.t, args.l

def main():
    mode, test, laptop = buildArgParser()
    print("laptop:", laptop)

    # Connect and start fly
    if not test:
        print("Working on real mode")
    else:
        print("Working on test mode")
    # Handle drone through chosen interface
    if mode == 'k':
        tCtrl = telloControl(test)
        print("Starting Tello control with keyboard...")
        tCtrl.keyControl()
    elif mode == 'c':
        tCtrl = telloControl(test)
        print("Starting Tello control with command...")
        tCtrl.commandControl()
    elif mode == 's':
        tCamera = telloCamera(test, trackfunction="Object", useDroneCamera=not laptop)
        tCamera.calibrate()
    elif mode == 'p':
        try:
            tCamera = telloCamera(test, useDroneCamera=not laptop)
            tCamera.startVideoLoopSearchShapes()
        except ValueError:
            pass
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif mode == 'obj':
        print("Following object...")
        tCamera = telloCamera(test, "Object", not laptop)
        tCamera.startVideoLoopTarget()
    elif mode == 'face':
        print("Following face...")
        tCamera = telloCamera(test, "Face", not laptop)
        tCamera.startVideoLoopTarget()
    elif mode == 'r':
        print("Recognizing hand numbers...")
        tCamera = telloCamera(test, useDroneCamera=not laptop)
        tCamera.startVideoLoopSearchHand()

if __name__ == '__main__':
    exit(main())

