from definitions import ROOT_DIR


class Worker:

    def __init__(self, path):
        self.file = ROOT_DIR + path
        self.model = None

    def predict(self, image_generator):
        raise NotImplementedError
        pass

    def train(self, train_generator=None, validation_generator=None):
        raise NotImplementedError
        pass

    def buildModel(self):
        raise NotImplementedError
        pass