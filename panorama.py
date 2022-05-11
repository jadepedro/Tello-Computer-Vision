import cv2
import numpy as np
from utils_mios import stackImages


class panorama(object):
    """
    Builds a panorama image from camera. Two approaches are implemented in this class.
    1. Using built in Stitcher function. Implemented with capturePanorama and processPanorama
    2. SIFT based  (see https://medium.com/@navekshasood/image-stitching-to-create-a-panorama-5e030ecc8f7)
    """
    # panoramed image
    m_pan_img = None

    # cv2 stitcher
    m_stitcher = cv2.Stitcher.create(cv2.STITCHER_SCANS)

    # image array
    m_img_array = []

    # pan array: there can be several blocks with different faces
    m_pan_array = []

    # face detector
    m_face_detector = None

    def __init__(self, face_detector):
        self.m_face_detector = face_detector

    def capturePanorama(self, frame, reset=False, show=True):
        # reset if requested
        if reset:
            self.m_img_array = []
        # look for faces
        #num_faces = self.m_face_detector.detectFaces(frame, True)
        #if num_faces > 0:
        if True:
            # add frame to array
            self.m_img_array.append(frame)
            # show if requested
            if show:
                l = len(self.m_img_array)
                #cv2.imshow('frame' + str(l), frame)
                cv2.imshow('frame', frame)
        print("Number of pictures: " + str(len(self.m_img_array)))

    def processPanorama(self, stitch=True):
        try:
            print("process Panorama - image count: " + str(len(self.m_img_array)))
            # call stitcher for certain number of new images

            if stitch:
                result, output = self.m_stitcher.stitch(self.m_img_array)
                #result, output = self.m_stitcher.stitch(imgs)
                if result != cv2.STITCHER_OK:
                    print("Failure to stitch: "+str(result))

                else:
                    print("Stitch ok: " + str(result))
                    self.m_pan_img = output
                #cv2.imwrite('d:\panorama.jpg',output)

            else:
                self.m_pan_img = stackImages(0.5, self.m_img_array)
        except ValueError:
            if not self.m_tello is None:
                self.m_tello.land()
        return self.m_pan_img

    def extendPanorama(self, img):
        # convert to grayscale
        img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # if this is the first img, no need to process, just add
        if self.m_pan_img is None:
            self.m_pan_img = img_grey
        else:
            # compute SIFT keypoints
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(self.m_pan_img, None)
            kp2, des2 = sift.detectAndCompute(img_grey, None)
            # find top M matches
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m in matches:
                if (m[0].distance < 0.5 * m[1].distance):
                    good.append(m)
            matches = np.asarray(good)

            # align using homograpyh trasnformation
            try:
                if (len(matches[:, 0]) >= 4):
                    src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
                    dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
                    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                else:
                    raise AssertionError('Can’t find enough keypoints.')


                # stitch the images
                dst = cv2.warpPerspective(self.m_pan_img, H, ((self.m_pan_img.shape[1] + img_grey.shape[1]), img_grey.shape[0]))  # wraped image
                dst[0:img_grey.shape[0], 0:img_grey.shape[1]] = img_grey  # stitched image
                self.m_pan_img = dst
                cv2.imshow('SIFT pan', dst)
                cv2.waitKey(1)

            except:
                pass




