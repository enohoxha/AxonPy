import ntpath
from threading import Thread

import cv2 as cv
import numpy as np

import definitions


def getFaceFromImage(image):
    if isinstance(image, str):
        img = cv.imread(image)
    else:
        img = image
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    haar_classifier = cv.CascadeClassifier('/home/manoolia/anaconda3/envs/axonpy/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    faces = haar_classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    return faces


def getFileNameFromPath(fileName):
    ntpath.basename(definitions.ROOT_DIR + "/outputs/")
    head, tail = ntpath.split(fileName)
    return tail


def writeImageText(image, text):
    font = cv.FONT_HERSHEY_SIMPLEX
    # get boundary of this text
    textsize = cv.getTextSize(text, font, 1, 2)[0]
    # get coords based on boundary
    textX = (image.shape[1] - textsize[0]) // 2
    textY = (image.shape[0] + textsize[1]) // 2 + (image.shape[0] // 2 - 20)

    # add text centered on image
    cv.putText(
        image,
        text,
        (textX, textY),
        font,
        1,
        (0, 0, 255),
        3
    )

    return image


def write_bb_image(boxes, labels, width_scale, height_scale, image, classes):


    if len(boxes) > 0:
        for (left, top, right, bottom), label in zip(boxes, labels):
            left = int(left * width_scale)
            top = int(top * height_scale)
            right = int(right * width_scale)
            bottom = int(bottom * height_scale)

            cv.rectangle(image, (left, top), (right, bottom), color=(255, 0, 0), thickness=3)
            cv.putText(
                image, classes[label], (left, top - 10),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0)
            )

    return image


class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.video_show = VideoShow(self.frame).start()

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
                self.video_show.frame = self.frame


    def stop(self):
        self.stopped = True



class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            cv.imshow("Video", self.frame)
            if cv.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True
