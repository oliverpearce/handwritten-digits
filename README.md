# handwritten-digits
the "hello world" of ai!

## set-up
running ```python3 -m venv .venv``` creates a [virtual environment](https://docs.python.org/3/library/venv.html) to install packages

from there, using ```pip install [package-name]``` allows you to download the package into the virtual environment!

## dataset
the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is the standard for handwritten digit recognition, which can be retrieved using [keras](https://keras.io/api/datasets/mnist/). this dataset consists of training and test data from which we can train our model. 

## vizualization
in order to vizualize the dataset, we use the [matplotlib library](https://matplotlib.org/stable/users/index).