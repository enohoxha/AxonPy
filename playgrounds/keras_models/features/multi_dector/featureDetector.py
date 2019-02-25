from time import time
import numpy as np
from scipy.misc import imresize

import definitions
from playgrounds.core.features import Feature
from playgrounds.keras_models.features.multi_dector.workers import tiny_yolo_v2
import cv2

from playgrounds.utilities import opencv_utilities


class FeatureDetector(Feature):

    def __init__(self):
        super().__init__()
        self.workers = {
            "tiny_yolo_v2": tiny_yolo_v2
        }

    def runFeature(self, worker, inputData, inType="img"):
        classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        assert worker in self.workers, "Worker do not exist plz change worker name"

        if inType == "video":
            self.videoDetection(classes, inputData, worker)
        if inType == "img":
            self.imageDetector(worker, inputData, classes)


    def videoDetection(self, classes, inputData, worker):
        IM_SIZE = 416
        webcam = cv2.VideoCapture(inputData)
        disp_height, disp_width = webcam.read()[1].shape[:2]
        height_scale = disp_height / IM_SIZE
        width_scale = disp_width / IM_SIZE
        n_frame = 1
        net = self.workers[worker].TinyYOLOv2(IM_SIZE, 5, 20)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        path = definitions.ROOT_DIR + "/outputs/tiny_yolo/" + opencv_utilities.getFileNameFromPath(inputData)
        out = cv2.VideoWriter(path, fourcc, 20.0, (854, 480))
        while True:
            ok, frame = webcam.read()
            if not ok:
                break
            if type(frame) is np.ndarray:
                net_frame = imresize(frame, (IM_SIZE, IM_SIZE)) / 255
                if n_frame == 1 or n_frame % 3 == 0:
                    boxes, labels = net.predict(net_frame)[0]

                if len(boxes) > 0:
                    for (left, top, right, bottom), label in zip(boxes, labels):
                        left = int(left * width_scale)
                        top = int(top * height_scale)
                        right = int(right * width_scale)
                        bottom = int(bottom * height_scale)

                        cv2.rectangle(frame, (left, top), (right, bottom), color=(255, 0, 0), thickness=3)
                        cv2.putText(
                            frame,
                            classes[label],
                            (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255)
                        )

                out.write(frame)
                cv2.imshow('frame', frame)
                n_frame += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        out.release()
        cv2.destroyAllWindows()



    def imageDetector(self, worker, inputData, classes):
        IMAGE_SIZE = 416
        boxes, labels = self.workers[worker].TinyYOLOv2(IMAGE_SIZE, 5, 20).predict(
            imresize(cv2.imread(inputData), (IMAGE_SIZE, IMAGE_SIZE)) / 255)[0]

        disp_height, disp_width, c = cv2.imread(inputData).shape

        height_scale = disp_height / IMAGE_SIZE
        width_scale = disp_width / IMAGE_SIZE

        saveImage = opencv_utilities.write_bb_image(boxes, labels, width_scale, height_scale, cv2.imread(inputData),
                                                    classes)
        cv2.imwrite(definitions.ROOT_DIR + "/outputs/tiny_yolo/" + opencv_utilities.getFileNameFromPath(inputData),
                    saveImage)


    def trainFeature(self, worker):
        IMAGE_SIZE = 416
        assert worker in self.workers, "Worker do not exist plz change worker name"
        self.workers[worker].TinyYOLOv2(IMAGE_SIZE, 5, 20).printModel()
