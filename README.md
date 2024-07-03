# handwritten-digits
the "hello world" of ai!

## usage
running ```python3 -m venv .venv``` creates a [virtual environment](https://docs.python.org/3/library/venv.html) to install packages locally in reference to the project.

from there, using ```pip install [package-name]``` allows you to download the necessary packages into the virtual environment!

## requirements
- python (3.12.4)
- keras (3.3.3)
    - tensorflow (2.16.1)
- matplotlib (3.9.0)
- pygame (2.6.0)
- opencv-python (4.10.0.84)
- ... more to come

## dataset
the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is the standard for handwritten digit recognition, which can be retrieved using [keras](https://keras.io/api/datasets/mnist/). this dataset consists of training and test data from which we can train our model. 

## vizualization
in order to vizualize the dataset, we use the [matplotlib library](https://matplotlib.org/stable/users/index). it is commented out by default, but can be easily used to display images from the MNIST dataset. 