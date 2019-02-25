import os
import sys
from threading import Thread

import cv2
import pyttsx3

import definitions
from playgrounds.core.workers import Worker
import bz2
import numpy as np
from keras.layers import Input

from playgrounds.keras_models.features.face_recognition.workers.facenet.align import align_image, AlignDlib
from playgrounds.keras_models.features.face_recognition.workers.facenet.facenet import create_model
from urllib.request import urlopen

from playgrounds.keras_models.features.face_recognition.workers.facenet.facenet_utils import IdentityMetadata


class FaceNetWorker(Worker):

    def __init__(self):
        # Input for anchor, positive and negative images
        self.in_a = Input(shape=(96, 96, 3))
        self.in_p = Input(shape=(96, 96, 3))
        self.in_n = Input(shape=(96, 96, 3))
        self.emb_a = None
        self.emb_p = None
        self.emb_n = None
        self.database = []
        self.image_generator = None
        self.alignment = AlignDlib(definitions.ROOT_DIR + '/trained_models/keras/dlib/landmarks.dat')
        self.stopped = False


        super().__init__("/trained_models/keras/facenet/nn4.small2.v1.h5")

    def buildModel(self):
        # Output for anchor, positive and negative embedding vectors
        # The nn4_small model instance is shared (Siamese network)
        self.model = create_model()
        self.model.load_weights(self.file)
        self.emb_a = self.model(self.in_a)
        self.emb_p = self.model(self.in_p)
        self.emb_n = self.model(self.in_n)


    def loadDataset(self, path):
        for i in os.listdir(path):
            for f in os.listdir(os.path.join(path, i)):
                # Check file extension. Allow only jpg/jpeg' files.
                ext = os.path.splitext(f)[1]
                if ext == '.jpg' or ext == '.jpeg':
                    img = self.load_image(os.path.join(path, i, f))
                    img = align_image(img, self.alignment)
                    # scale RGB values to interval [0,1]
                    img = (img / 255.).astype(np.float32)
                    # obtain embedding vector for image
                    self.database.append(IdentityMetadata(path, i, f, self.model.predict(np.expand_dims(img, axis=0))[0]))


    def predicts(self):
        s = ""
        while True and not self.stopped:
            image = self.image_generator[..., ::-1]
            image = align_image(image, self.alignment)
            self.database = np.array(self.database)
            name = "Mehhhhh no people here."

            if image is not None:
                readyToPredict = True
            else:
                readyToPredict = False

            if readyToPredict:
                name = "I don't know this person"
                image = (image / 255.).astype(np.float32)
                current_image = self.model.predict(np.expand_dims(image, axis=0))[0]
                for i, m in enumerate(self.database):
                    thresh = self.distance(current_image, m.prediction)
                    if thresh < 0.5:
                        name = "This person is " + m.name
                        break

            print("\r" + name, end=" ", flush=True)

            if s != name:
                s = name

                # self.engine.say(name)
                # self.engine.runAndWait()

    def start(self):
        Thread(target=self.predicts, args=()).start()
        return self


    @staticmethod
    def distance(emb1, emb2):
        return np.sum(np.square(emb1 - emb2))

    @staticmethod
    def load_image(path):
        img = cv2.imread(path)
        # OpenCV loads images with color channels
        # in BGR order. So we need to reverse them
        return img[..., ::-1]


    @staticmethod
    def download_landmarks(dst_file):
        url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
        decompressor = bz2.BZ2Decompressor()

        with urlopen(url) as src, open(dst_file, 'wb') as dst:
            data = src.read(1024)
            while len(data) > 0:
                dst.write(decompressor.decompress(data))
                data = src.read(1024)

    def show_pair(self, idx1, idx2):

        print(self.distance(self.database[idx1].prediction, self.database[idx2].prediction))


    def stopThread(self):
        self.stopped = True
