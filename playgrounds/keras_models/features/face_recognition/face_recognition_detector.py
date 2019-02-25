import cv2

from playgrounds.core.features import Feature
from playgrounds.keras_models.features.face_recognition.workers.facenet.facenet_worker import FaceNetWorker
import definitions

from playgrounds.utilities.opencv_utilities import VideoGet


class FaceRecognition(Feature):

    def __init__(self):
        super().__init__()
        self.workers = {
            "FaceNet": FaceNetWorker()
        }
        self.database = {}


    def runFeature(self, worker, inputData, inType ="image"):
        self.worker = self.workers.get(worker)
        func = self.inputTypes.get(inType)
        func(worker, inputData)


    def runOnLiveStream(self, worker, inputData):

        self.worker = self.workers.get(worker)
        self.worker.buildModel()
        # self.worker.download_landmarks(definitions.ROOT_DIR + '/trained_models/keras/dlib/landmarks.dat')
        self.worker.loadDataset(definitions.ROOT_DIR + "/datasets/facenet_db")
        video_getter = VideoGet(0).start()
        self.worker.image_generator = video_getter.frame
        w = self.worker.start()

        while True:
            if video_getter.stopped:
                video_getter.stop()
                break
            frame = video_getter.frame
            if frame is not None:
                w.image_generator = frame


    def runOnImage(self, worker, inputData):
        self.worker = self.workers.get(worker)
        self.worker.buildModel()
        self.worker.loadDataset(definitions.ROOT_DIR + "/datasets/facenet_db")
        self.worker.image_generator = cv2.imread(inputData)
        self.worker.start()
        self.worker.stopThread()


