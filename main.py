import pygame, sys
from pygame.locals import *
import numpy as np
from keras.api.models import load_model
import cv2

# constants
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

# initialize pygame window
pygame.init()

# configure pygame display
pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("window of doom")

# forever loop for display processing
while 1:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

