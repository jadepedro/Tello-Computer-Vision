import cv2

class panorama(object):
    """
    Builds a panorama image from camera
    """
    # panoramed imaga
    m_pan_img = None

    # cv2 stitcher
    m_stitcher = cv2.Stitcher.create()

    # image array
    m_img_array = []

    def __init__(self):
        pass

    def updatePanorama(self, frame):
        try:
            # add frame to array and try to stitch
            self.m_img_array.append(frame)
            if self.m_pan_img is None:
                self.m_pan_img = frame
            print("image count: " + str(len(self.m_img_array)))
            # call stitcher
            result, output = self.m_stitcher.stitch(self.m_img_array)
            #result, output = self.m_stitcher.stitch(imgs)
            if result != cv2.STITCHER_OK:
                print("Failure to stitch: "+str(result))
            else:
                self.m_pan_img = output
        except ValueError:
            if not self.m_tello is None:
                self.m_tello.land()
        return self.m_pan_img


