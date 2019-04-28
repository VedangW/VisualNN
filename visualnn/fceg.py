import keras_loader
import fc
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense

# vnn.init_notebook_mode(connected=False)
model = keras_loader.loadFromFile('./mlp1.h5')
model.plot()
model2 = keras_loader.loadFromJSON('./mlp3.json')
model2.plot()
model3 = keras_loader.loadFromFile('./mlp2.h5')
model3.plot()

model4 = Sequential([
    Dense(2, input_shape=(3,)),
    Activation('relu'),
    Dense(1),
    Activation('relu'),
    Dense(5),
    Activation('softmax')
])

model4 = keras_loader.kerasToVnn(model4)