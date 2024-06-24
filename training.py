# for vizualization and data processing
import numpy as np
import matplotlib.pyplot as plt

# for retrieving the dataset
import keras
from keras.datasets import mnist

# get data and preprocess it
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# vizualize dataset
def plot_input_img(i):
    plt.imshow(X_train[i], cmap='binary')
    plt.title(y_train[i])
    plt.show()

for i in range(2):
    plot_input_img(i)

# pre process the image
X_train = X_train.astype(np.float32)/255 #normalize the image to [0, 1]
X_test = X_test.astype(np.float32)/255

X_train = np.expand_dims(X_train, -1) # expand images to (28, 28, 1) for number predictor feature
X_test = np.expand_dims(X_test, -1)

print(X_train.shape)
print(X_test.shape)
