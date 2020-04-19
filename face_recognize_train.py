
import numpy as np # forlinear algebra
import matplotlib.pyplot as plt

import keras

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications.vgg16 import VGG16
from keras.models import Model

import os


train_folder= 'data/train/'
val_folder = 'data/val/'

class_array = os.listdir(train_folder) # dir is your directory path
num_class = len(class_array)
vgg = VGG16(input_shape=[150,150] + [3], weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False
x = Flatten()(vgg.output)
prediction = Dense(num_class, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)  #Image normalization.

training_set = train_datagen.flow_from_directory(train_folder,
                                                 target_size = (150,150),
                                                 color_mode="rgb",
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

validation_generator = test_datagen.flow_from_directory(val_folder,
    target_size=(150,150),
    color_mode="rgb",
    batch_size=32,
    class_mode='categorical')

model.summary()

model.fit_generator(training_set,
                         steps_per_epoch = 100,
                         epochs = 5,
                         validation_data = validation_generator,
                         validation_steps = 624)
model.save('face_recognizer.h5')