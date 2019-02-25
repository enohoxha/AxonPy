from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, MaxPooling2D, Activation, Dropout, Flatten
import os
import numpy as np
from playgrounds.core.workers import Worker
from keras import applications
import definitions


def getCallbacks(file):
    callbacks = [
        ModelCheckpoint(file, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
        EarlyStopping(monitor='val_acc', patience=4, verbose=1)
    ]
    return callbacks


class DogCatWorkers (Worker):

    def __init__(self):
        super().__init__("/trained_models/keras/dog_cat/worker1_weights.h5")

    def predict(self, image_generator):
        return self.model.predict(image_generator)

    def buildModel(self):

        if os.path.exists(self.file):
            self.model = load_model(self.file)
        else:
            self.model = Sequential();
            self.model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            self.model.add(Conv2D(32, (3, 3)))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            self.model.add(Conv2D(64, (3, 3)))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
            self.model.add(Dense(64))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(1))
            self.model.add(Activation('sigmoid'))
            self.model.compile(loss='binary_crossentropy',
                               optimizer='rmsprop',
                               metrics=['accuracy'])
        return self



    def train(self, train_generator=None, validation_generator=None):
        batch_size = 16

        self.model.fit_generator(
            train_generator,
            steps_per_epoch=2000 // batch_size,
            epochs=10,
            validation_data=validation_generator,
            validation_steps=800 // batch_size,
            callbacks=getCallbacks(self.file)
        )

    def printModel(self):
        print(self.model.summary())

    @staticmethod
    def getSpecifications():
        return 'binary'


class DogCatVGG16 (Worker):

    def __init__(self):
        super().__init__("/trained_models/keras/dog_cat/workerVGG_weights.h5")
        self.img_width, self.img_height = 150, 150


        self.epochs = 50
        self.batch_size = 16

    def buildModel(self):
        if os.path.exists(definitions.ROOT_DIR + '/trained_models/keras/dog_cat/bottleneck_features_train.npy'):

            if os.path.exists(self.file):
                self.model = load_model(self.file)

            else:
                train_data = np.load(definitions.ROOT_DIR + '/trained_models/keras/dog_cat/bottleneck_features_train.npy')

                self.model = Sequential()
                self.model.add(Flatten(input_shape=train_data.shape[1:]))
                self.model.add(Dense(256, activation='relu'))
                self.model.add(Dropout(0.5))
                self.model.add(Dense(1, activation='sigmoid'))
                self.model.compile(
                    optimizer='rmsprop',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
        else:
            self.model = applications.VGG16(include_top=False, weights='imagenet')

        return self


    def predict(self, image_generator):
        temp_model = applications.VGG16(include_top=False, weights='imagenet')
        img = temp_model.predict(image_generator)
        return self.model.predict(img)

    def train(self, train_generator=None, validation_generator=None):

        if os.path.exists(definitions.ROOT_DIR + '/trained_models/keras/dog_cat/bottleneck_features_train.npy'):

            train_data = np.load(definitions.ROOT_DIR + '/trained_models/keras/dog_cat/bottleneck_features_train.npy')

            train_labels = np.array([0] * 1000 + [1] * 1000)

            validation_data = np.load(definitions.ROOT_DIR + '/trained_models/keras/dog_cat/bottleneck_features_validation.npy')

            validation_labels = np.array([0] * 400 + [1] * 400)

            self.model.fit(
                train_data,
                train_labels,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(validation_data, validation_labels),
                callbacks=getCallbacks(self.file)
            )

        else:
            self.loadTrainData(train_generator, validation_generator)

    def loadTrainData(self, train_generator=None, validation_generator=None):
            bottleneck_features_train = self.model.predict_generator(train_generator, 2000 // self.batch_size,  verbose=1)

            np.save(definitions.ROOT_DIR + '/trained_models/keras/dog_cat/bottleneck_features_train.npy',
                    bottleneck_features_train)

            bottleneck_features_validation = self.model.predict_generator(
                validation_generator, 800 // self.batch_size, verbose=1)

            np.save(definitions.ROOT_DIR + '/trained_models/keras/dog_cat/bottleneck_features_validation.npy',
                    bottleneck_features_validation)

    @staticmethod
    def getSpecifications():
        return None
