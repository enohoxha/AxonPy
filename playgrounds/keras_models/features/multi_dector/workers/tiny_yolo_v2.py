import os

import IPython
import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Reshape
from keras import backend as K
from keras.utils import plot_model

import definitions
from keras import Model
import numpy as np

from playgrounds.core.workers import Worker
from playgrounds.utilities.dark_net_utilities import load_weights
from playgrounds.utilities.YOLOUtilities import yoloPostProcess
from playgrounds.utilities.YOLOUtilities import conv_batch_lrelu

TINY_YOLOV2_ANCHOR_PRIORS = np.array([
    1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52
]).reshape(5, 2)


class TinyYOLOv2(Worker):

    def __init__(self, image_size, B, n_classes, is_learning_phase=False):
        """

        :param image_size: size of input image
        :param B: Number of boundary boxes in each cell grid
        :param n_classes: number of classes to predict
        :param is_learning_phase: default false
        """
        path = ""


        super().__init__(path)
        K.set_learning_phase(int(is_learning_phase))
        K.reset_uids()

        self.image_size = image_size
        self.n_cells = self.image_size // 32
        self.B = B
        self.n_classes = n_classes

        self.model = self.buildModel()

        if os.path.exists(definitions.ROOT_DIR + "/trained_models/keras/yolov2/yolov2-tiny-voc-keras.weights"):
            self.file = definitions.ROOT_DIR + "/trained_models/keras/yolov2/yolov2-tiny-voc-keras.weights"
            self.loadWeightsFromKeras()

        else:
            self.file = definitions.ROOT_DIR + "/trained_models/keras/yolov2/yolov2-tiny-voc.weights"
            self.loadWeightsFromDarknet()



    def buildModel(self):

        model_in = Input((self.image_size, self.image_size, 3))

        model = model_in

        for i in range(0, 5):
            model = conv_batch_lrelu(model, 16 * 2 ** i, 3)
            model = MaxPooling2D(2, padding='valid')(model)

        model = conv_batch_lrelu(model, 512, 3)
        model = MaxPooling2D(2, 1, padding='same')(model)

        model = conv_batch_lrelu(model, 1024, 3)
        model = conv_batch_lrelu(model, 1024, 3)

        model = Conv2D(125, (1, 1), padding='same', activation='linear')(model)

        model_out = Reshape(
            [self.n_cells, self.n_cells, self.B, 4 + 1 + self.n_classes]
        )(model)

        return Model(inputs=model_in, outputs=model_out)


    def loadWeightsFromKeras(self):
        self.model.load_weights(self.file)

    def loadWeightsFromDarknet(self):
        load_weights(self.model, self.file)
        self.model.save(definitions.ROOT_DIR + "/trained_models/keras/yolov2/yolov2-tiny-voc-keras.weights")

    def printModel(self):
        keras.utils.plot_model(self.model, to_file='test_keras_plot_model.png', show_shapes = True)

        print(self.model.summary())

    def predict(self, images):
        if len(images.shape) == 3:
            # single image
            images = images[None]

        output = self.model.predict(images)
        return yoloPostProcess(output, TINY_YOLOV2_ANCHOR_PRIORS)
