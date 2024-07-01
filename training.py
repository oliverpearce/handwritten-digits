# for vizualization and data processing
import numpy as np
import matplotlib.pyplot as plt

# for retrieving the dataset
import keras
from keras.api.datasets import mnist

# for building model
from keras.api.models import Sequential
from keras.api.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout

# for callbacks
from keras.api.callbacks import EarlyStopping, ModelCheckpoint

# get data and preprocess it
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# vizualize dataset
# def plot_input_img(i):
#     plt.imshow(X_train[i], cmap='binary')
#     plt.title(y_train[i])
#     plt.show()

# for i in range(2):
#     plot_input_img(i)

# pre process the image
X_train = X_train.astype(np.float32)/255 #normalize the image to [0, 1]
X_test = X_test.astype(np.float32)/255

# expand images to (28, 28, 1) for number predictor feature
X_train = np.expand_dims(X_train, -1) 
X_test = np.expand_dims(X_test, -1)

# print(X_train.shape)
# print(X_test.shape)

# one hot encoding
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# print(y_train)
# print(y_test)

# create model
model = Sequential()

# add layers
model.add(Input(shape=(28,28,1)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D())

# flatten model
model.add(Flatten())

# dropout to prevent overfitting
model.add(Dropout(0.25))

# classification layer
model.add(Dense(10, activation='softmax'))

# compile model
model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

# callbacks!
es = EarlyStopping(monitor='val acc', min_delta=0.01, patience=4, verbose=1)
mc = ModelCheckpoint("./bestmodel.keras", monitor='val acc', verbose=1, save_best_only=True)
cb = [es, mc]

# model training!! (will do later, i am lazy)