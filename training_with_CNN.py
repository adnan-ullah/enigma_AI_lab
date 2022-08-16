from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

cf = Sequential()
cf.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
cf.add(MaxPooling2D(pool_size=(2, 2)))
cf.add(Flatten())
cf.add(Dense(units=128, activation='relu'))
cf.add(Dense(units=1, activation='sigmoid'))

cf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)
training_set = train_datagen.flow_from_directory('Cats and Dogs/training_set/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')
test_set = test_datagen.flow_from_directory('Cats and Dogs/test_set/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

cf.fit(training_set,
                 steps_per_epoch=250,
                 epochs=25,
                 validation_data=test_set,
                 validation_steps=65)

cf.save('finalTrained.h5')



