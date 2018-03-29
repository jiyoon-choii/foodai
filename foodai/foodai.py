#!/usr/bin/python
import sys, getopt, os, os.path
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense


def main(argv):
    imgWidth = 128
    imgHeight = 128
    topModelWeightsPath = 'bottleneck_fc_model.h5'
    trainDataDir = 'data/train'
    validationDataDir = 'data/validation'
    epochs = 50
    batchSize = 16
    outputFile = 'foodai_model.h5'
    try:
        opts, args = getopt.getopt(argv, "hi:o:", [
            "imgWidth=",
            "imgHeight=",
            "topModelWeightsPath=",
            "trainDataDir=",
            "validationDataDir=",
            "epochs=",
            "batchSize=",
            "outputFile="])
    except getopt.GetoptError:
        print (
            'usage: foodai.py --imgWidth=<imgWidth> --imgHeight=<imgHeight> --topModelWeightsPath=<topModelWeightsPath> --trainDataDir=<trainDataDir> --validationDataDir=<validationDataDir> --epochs=<epochs> --batchSize=<batchSize> --outputFile=<outputFile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print (
                'usage: foodai.py --imgWidth=<imgWidth> --imgHeight=<imgHeight> --topModelWeightsPath=<topModelWeightsPath> --trainDataDir=<trainDataDir> --validationDataDir=<validationDataDir> --epochs=<epochs> --batchSize=<batchSize> --outputFile=<outputFile>')
            sys.exit()
        elif opt in ("--imgWidth"):
            imgWidth = int(arg)
        elif opt in ("--imgHeight"):
            imgHeight = int(arg)
        elif opt in ("--topModelWeightsPath"):
            topModelWeightsPath = int(arg)
        elif opt in ("--trainDataDir"):
            trainDataDir = arg
        elif opt in ("--validationDataDir"):
            validationDataDir = arg
        elif opt in ("--epochs"):
            epochs = int(arg)
        elif opt in ("--batchSize"):
            batchSize = int(arg)
        elif opt in ("--outputFile"):
            outputFile = arg

    print ('imgWidth is ', imgWidth)
    print ('imgHeight is ', imgHeight)
    print ('topModelWeightsPath is ', topModelWeightsPath)
    print ('trainDataDir is ', trainDataDir)
    print ('validationDataDir is ', validationDataDir)
    print ('epochs is ', epochs)
    print ('batchSize is ', batchSize)
    print ('outputFile is ', outputFile)

    trainDataSize = 0
    for root, dirs, files in os.walk(trainDataDir):
        trainDataSize += len([file for file in files if file != '.DS_Store'])
    print ('Train data length', trainDataSize)

    validationDataSize = 0
    for root, dirs, files in os.walk(validationDataDir):
        validationDataSize += len([file for file in files if file != '.DS_Store'])
    print ('Validation data length', validationDataSize)

    initRecognition(imgWidth, imgHeight, topModelWeightsPath, trainDataDir, validationDataDir, epochs, batchSize,
                    trainDataSize, validationDataSize)
    # resolveRecognition(imgWidth, imgHeight, topModelWeightsPath, trainDataDir, validationDataDir, epochs, batchSize,
    #                    trainDataSize, validationDataSize, outputFile)


def resolveRecognition(imgWidth, imgHeight, topModelWeightsPath, trainDataDir, validationDataDir, epochs, batchSize,
                       trainDataSize, validationDataSize, outputFile):
    # build the VGG16 network
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(imgWidth, imgHeight, 3))
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights(topModelWeightsPath)

    # add the model on top of the convolutional base
    # model.add(top_model)
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        trainDataDir,
        target_size=(imgHeight, imgWidth),
        batch_size=batchSize,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validationDataDir,
        target_size=(imgHeight, imgWidth),
        batch_size=batchSize,
        class_mode='binary')

    # fine-tune the model
    model.fit_generator(
        train_generator,
        samples_per_epoch=trainDataSize,
        epochs=epochs,
        validation_data=validation_generator,
        nb_val_samples=validationDataSize)

    print('Save model...')
    model.save(outputFile)


def initRecognition(imgWidth, imgHeight, topModelWeightsPath, trainDataDir, validationDataDir, epochs, batchSize,
                    trainDataSize, validationDataSize):
    save_bottleneck_features(trainDataDir, validationDataDir, imgWidth, imgHeight, batchSize, trainDataSize,
                             validationDataSize)
    train_top_model(topModelWeightsPath, batchSize, trainDataSize, validationDataSize, epochs)
    print('init recognition done...')


def save_bottleneck_features(trainDataDir, validationDataDir, imgWidth, imgHeight, batchSize, trainDataSize,
                             validationDataSize):
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    print('Model loaded...')

    generator = datagen.flow_from_directory(
        trainDataDir,
        target_size=(imgWidth, imgHeight),
        batch_size=batchSize,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, trainDataSize // batchSize)
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validationDataDir,
        target_size=(imgWidth, imgHeight),
        batch_size=batchSize,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, validationDataSize // batchSize)
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
    print('Saved both bottleneck features...')


def train_top_model(topModelWeightsPath, batchSize, trainDataSize, validationDataSize, epochs):
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array([0] * (trainDataSize / 2) + [1] * (trainDataSize / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array([0] * (validationDataSize / 2) + [1] * (validationDataSize / 2))

    print('Data loaded...')

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batchSize,
              validation_data=(validation_data, validation_labels))
    model.save_weights(topModelWeightsPath)
    print('Saved weights...')


if __name__ == "__main__":
    main(sys.argv[1:])
