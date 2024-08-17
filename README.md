# handwritten-digits
the "hello world" of ai!

## set-up
running `python3 -m venv .venv` creates a [virtual environment](https://docs.python.org/3/library/venv.html) to install packages locally in reference to the project.

after that, running `pip install -r requirements.txt` allows you to download the necessary packages into the virtual environment!

## usage
the command `source .venv/bin/activate` activates the venv, from where you can run the main file with `python3 main.py`! any number drawn in one continuous stroke will be compared against the model, and the prediction will be shown. the command `deactivate` deactivates the venv.

## dataset
the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is the standard for handwritten digit recognition, which can be retrieved using [keras](https://keras.io/api/datasets/mnist/). this dataset consists of training and test data from which we can train our model. 

## vizualization
in order to vizualize the dataset, we use the [matplotlib library](https://matplotlib.org/stable/users/index). it is commented out by default, but can be easily used to display images from the MNIST dataset. 