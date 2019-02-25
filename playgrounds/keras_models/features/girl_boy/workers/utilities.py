from keras.preprocessing.image import ImageDataGenerator



def get_training_data_from_generator(train_path, test_path):
    train_gen = ImageDataGenerator(rescale=1. / 255,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rotation_range=10,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

    test_gen = ImageDataGenerator(rescale=1. / 255)
    train = train_gen.flow_from_directory(train_path,
                                          target_size=(64, 64),
                                          batch_size=16,
                                          class_mode='binary')

    test = test_gen.flow_from_directory(test_path,
                                        target_size=(64, 64),
                                        batch_size=16,
                                        class_mode='binary')
    return train, test



