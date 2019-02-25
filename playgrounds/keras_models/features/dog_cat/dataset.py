from definitions import ROOT_DIR
from keras.preprocessing.image import ImageDataGenerator

'''
In order to make the most of our few training examples,
we will "augment" them via a number of random transformations, so that our model would never see twice the exact same picture.
This helps prevent overfitting and helps the model generalize better.

'''


def generateImageFromDataset(classMode):
    batch_size = 16
    train_generator, validation_generator = None, None
    if classMode is not None:
        train_generator, validation_generator = generateDataCustom(batch_size=batch_size)
    else:
        train_generator, validation_generator = generateDataVGG16(batch_size=batch_size)

    return train_generator, validation_generator


def generateDataCustom(batch_size):
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        ROOT_DIR + '/datasets/dog_cat/data/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode="binary")  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for train_generatorvalidation data
    validation_generator = test_datagen.flow_from_directory(
        ROOT_DIR + '/datasets/dog_cat/data/validation',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode="binary")

    return train_generator, validation_generator


def generateDataVGG16(batch_size):
    print("data from vgg16")
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(
        ROOT_DIR + '/datasets/dog_cat/data/train',  # this is the target directory
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode=None,  # this means our generator will only yield batches of data, no labels
        shuffle=False)
    generator2 = datagen.flow_from_directory(
         ROOT_DIR + '/datasets/dog_cat/data/validation',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    return generator, generator2