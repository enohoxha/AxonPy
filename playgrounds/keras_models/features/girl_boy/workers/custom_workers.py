from keras.callbacks import EarlyStopping

from playgrounds.keras_models.features.girl_boy.workers.utilities import get_training_data_from_generator
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
from playgrounds.core.workers import Worker
from keras import Sequential


class CustomWorker1(Worker):

    def __init__(self):
        super().__init__("/trained_models/keras/custom/w.h5")

    def buildModel(self):

        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(64, 64, 3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())

        self.model.add(Flatten())

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model.load_weights(self.file)


    def train(self, train_generator=None, validation_generator=None):
        self.buildModel()
        train, test = get_training_data_from_generator("/home/manoolia/code/python/axonpy/datasets/data/train",
                                                       "/home/manoolia/code/python/axonpy/datasets/data/test")
        self.model.fit_generator(train,
                            steps_per_epoch=800,
                            epochs=200,
                            validation_data=test,
                            validation_steps=200, callbacks=[EarlyStopping(monitor="acc", patience=1)])


    def predict(self, image):
        return self.model.predict(image)