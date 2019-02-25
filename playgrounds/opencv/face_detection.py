import cv2 as cv
import numpy as np

import definitions
from playgrounds.utilities import opencv_utilities


def detectFaces(url):
    img = cv.imread(url)
    faces = opencv_utilities.getFaceFromImage(img)
    for (x, y, width, height) in faces:
        cv.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

    cv.imwrite(definitions.ROOT_DIR + "/outputs/" + opencv_utilities.getFileNameFromPath(url), img)


