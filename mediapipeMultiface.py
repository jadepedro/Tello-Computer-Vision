import cv2
import mediapipe as mp

from faceDetectionDrawUtilities import draw_detection
from faceDetectionDrawUtilities import DrawingSpec

class multifaceDetector(object):

    # mediapipe drawing utils
    #m_mp_drawing = mp.solutions.drawing_utils
    # load face detection model
    m_mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=1
        ,  # model selection
        min_detection_confidence=0.5  # confidence threshold
    )

    def __init__(self):
        pass

    def detectFaces(self, img, draw=True, show=False, center_margin=0.10):
        # convert color scheme to mediapipe
        img_input = img
        #img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # get results
        results = self.m_mp_face.process(img_input)
        num_faces = 0
        num_faces_marked = 0
        if not results.detections:
            #print('No faces detected.')
            pass
        else:
            num_faces = len(results.detections)

            #print("Detected " + str(num_faces) + " faces")
            if draw:
                for detection in results.detections:  # iterate over each detection and draw on image
                    #self.m_mp_drawing.draw_detection(img, detection)
                    marker_applied=draw_detection(img, detection, DrawingSpec(), DrawingSpec(),
                                                  True,only_centered=True,
                                                  center_margin=center_margin)
                    if marker_applied:
                        num_faces_marked += 1
        if show:
            cv2.imshow('multiface', img_input)
        return num_faces, num_faces_marked
