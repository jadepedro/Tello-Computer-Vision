import numpy as np

class telloPIDControl:

    m_tello = None

    m_forw_back_vel = 0
    m_left_right_vel = 0
    m_yaw_vel = 0
    m_up_down_vel = 0

    m_test = True
    m_fly = True

    # PID hyper parameters (P, I, D) for boxed object track
    m_PID_hyp_boxed = np.array([0.4, 0.0, 0.5])
    # previous error boxed
    m_prevError_boxed = np.array([0, 0, 0])

    # size of image
    m_frameSize = np.array([320, 240])

    # target values for boxed
    m_center_boxed = np.array([160, 120, 6000])

    # previous error integra for boxed
    m_integralError_boxed = np.array([0., 0., 0.])

    # ratios to convert from pure error measure to the desired order
    m_yaw_ratio = 0.25
    m_up_down_ratio = 0.25
    m_forw_back_ratio = 0.003

    def __init__(self, tello, size, test, targetsize=6000):
        self.m_tello = tello
        self.m_frameSize = size
        self.m_center = np.array([int(size[0]/2), int(size[1]/2)])
        self.m_test = test
        self.m_fly = False
        self.m_center_boxed[2] = targetsize
        # print("fly: ",  self.m_fly)

        if test:
            status = self.m_tello.get_current_state()
            for prop in status:
                print(prop, status[prop])

    def setFly(self, fly):
        self.m_fly = fly

    def trackTargetBoxed(self, targetCoordinates, box_size):
        """
        Tracks boxed (includes bounding box object info) target through PID control
        :param targetCoordinates: current coordinates of the face
        :param box_size: current size of the bounding box
        :param fly: boolean to indicate if dron is currently flying or not (test mode)
        :return:
        """
        # PID
        kp = self.m_PID_hyp_boxed[0]
        ki = self.m_PID_hyp_boxed[1]
        kd = self.m_PID_hyp_boxed[2]

        # Create array with coordinates and box_size
        currentCoordinates_boxed = np.array([targetCoordinates[0], targetCoordinates[1], box_size])

        # current error x, y and box size error
        # if the target is not found set error to 0 to avoid movement
        if (targetCoordinates == np.array([0, 0])).all():
            error = np.array([0, 0, 0])
            difError = np.array([0, 0, 0])
        else:
            # use the array that includes box size target
            error = currentCoordinates_boxed - self.m_center_boxed
            # differential of error
            difError = error - self.m_prevError_boxed

        speed = kp*error + ki*self.m_integralError_boxed + kd*difError

        # limit the parameters as a security measure
        self.m_yaw_vel = np.clip(int(speed[0] * self.m_yaw_ratio), -100, 100)
        self.m_up_down_vel = np.clip(int(speed[1] * self.m_up_down_ratio), -100, 100) #+: up, -:down
        self.m_forw_back_vel = np.clip(int(speed[2] * self.m_forw_back_ratio), -100, 100) #+: back, -:fwd

        # Execute control
        #print("Error:", error, "difError:", difError, "integral error:", self.m_integralError_boxed)
        #print("Speed:", speed)
        print("yaw:", self.m_yaw_vel, " up_down:", self.m_up_down_vel, " fwd_back:", self.m_forw_back_vel)

        if self.m_fly:
            self.m_tello.send_rc_control(0, int(-self.m_forw_back_vel), int(-self.m_up_down_vel), int(self.m_yaw_vel))

        # update next values
        self.m_prevError_boxed = error
        # update integral error
        self.m_integralError_boxed += self.m_prevError_boxed


