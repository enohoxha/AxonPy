
class Feature:

    workers = {}
    inputTypes = {}

    def __init__(self):
        self.inputTypes = {
            'video': self.runOnVideo,
            'image': self.runOnImage,
            'livestream': self.runOnLiveStream,
        }
        self.worker = None
        self.workers = {}
        self.input = None

    def runFeature(self, worker, inputData, inType ="image"):
        assert worker in self.workers, "Worker do not exist plz change worker name"
        assert inType in self.inputTypes, "Input type is not known"

    def runOnVideo(self, worker, inputData):
        raise NotImplementedError
        pass

    def runOnImage(self, worker, inputData):
        raise NotImplementedError
        pass

    def runOnLiveStream(self, worker, inputData):
        raise NotImplementedError
        pass

    def train(self, worker, inputData):
        raise NotImplementedError
        pass

    def trainFeature(self, worker):
        raise NotImplementedError
        pass
