#!/usr/bin/python

import numpy as np
import sys, getopt, int
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense

# dimensions of our images.
img_width, img_height = 128, 128

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2800
nb_validation_samples = 800
epochs = 50
batch_size = 16

def main(argv):
    imgWidth = ''
    outputfile = ''
    topModelWeightsPath = ''
    trainDataDir = ''
    validationDataDir = ''
    epochs = ''
    batchSize = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",[
            "imgWidth=",
            "imgHeight=",
            "topModelWeightsPath=",
            "trainDataDir=",
            "validationDataDir=",
            "epochs=",
            "batchSize="])
    except getopt.GetoptError:
        print 'foodai_alt.py ' \
              '--imgWidth=<imgWidth> ' \
              '--imgHeight=<imgHeight> ' \
              '--topModelWeightsPath=<topModelWeightsPath> ' \
              '--trainDataDir=<trainDataDir> ' \
              '--validationDataDir=<validationDataDir> ' \
              '--epochs=<epochs> ' \
              '--batchSize=<batchSize>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            # print 'foodai_alt.py -i <inputfile> -o <outputfile>'
            print 'foodai_alt.py ' \
                  '--imgWidth=<imgWidth> ' \
                  '--imgHeight=<imgHeight> ' \
                  '--topModelWeightsPath=<topModelWeightsPath> ' \
                  '--trainDataDir=<trainDataDir> ' \
                  '--validationDataDir=<validationDataDir> ' \
                  '--epochs=<epochs> ' \
                  '--batchSize=<batchSize>'
            sys.exit()
        elif opt in ("--imgWidth"):
            imgWidth = int(arg)
            print 'imgWidth is "', imgWidth
        elif opt in ("--imgHeight"):
            imgHeight = int(arg)
            print 'imgHeight is "', imgHeight
        elif opt in ("--topModelWeightsPath"):
            topModelWeightsPath = int(arg)
            print 'topModelWeightsPath is "', topModelWeightsPath
        elif opt in ("--trainDataDir"):
            trainDataDir = arg
            print 'trainDataDir is "', trainDataDir
        elif opt in ("--validationDataDir"):
            validationDataDir = arg
            print 'validationDataDir is "', validationDataDir
        elif opt in ("--epochs"):
            epochs = int(arg)
            print 'epochs is "', epochs
        elif opt in ("--batchSize"):
            batchSize = int(arg)
            print 'batchSize is "', batchSize






def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array([0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array([0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


save_bottlebeck_features()
train_top_model()
###############
##
###############
# path to the model weights files.
weights_path = './downloads/vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 128, 128

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2800
nb_validation_samples = 800
epochs = 50
batch_size = 16

# build the VGG16 network
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
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
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
#model.add(top_model)
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
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

model.save('data/foodai_model.h5')

if __name__ == "__main__":
    main(sys.args[1:])