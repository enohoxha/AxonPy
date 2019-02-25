import cv2
import numpy as np
from playgrounds.keras_models.features.dog_cat import dataset, workers
from playgrounds.core.features import Feature
from playgrounds.utilities import opencv_utilities
import definitions


class DogCatFeature(Feature):

    def __init__(self):
        super().__init__()
        self.workers = {
            "c1": workers.DogCatWorkers().buildModel(),
            "VGG16": workers.DogCatVGG16().buildModel()
        }


    def runFeature(self, worker, inputData,  inType ="image"):
        assert worker in self.workers, "Worker do not exist plz change worker name"
        image = cv2.imread(inputData)
        resize_image = cv2.resize(image, (150, 150))
        resize_image = resize_image / 255
        resize_image = np.expand_dims(resize_image, axis=0)
        out = self.workers[worker].predict(resize_image)
        text = "Cat"
        if out[0][0] > 0.5:
            text = "Dog"
        saveImage = opencv_utilities.writeImageText(image, text)
        cv2.imwrite(definitions.ROOT_DIR + "/outputs/cat_dog/" + opencv_utilities.getFileNameFromPath(inputData), saveImage)


    def trainFeature(self, worker):
        assert worker in self.workers, "Worker do not exist plz change worker name"
        print(self.workers)
        train_generator, validation_generator = dataset.generateImageFromDataset(self.workers[worker].getSpecifications())

        self.workers[worker].train(train_generator, validation_generator)






