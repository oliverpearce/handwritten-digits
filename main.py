import pygame, sys
from pygame.locals import *
import numpy as np
from keras.api.models import load_model
import cv2

# constants
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

MODEL = load_model("bestmodel.keras")
LABELS = {0:"Zero", 1:"One", 2:"Two", 3:"Three", 4:"Four", 5:"Five", 6:"Six", 7:"Seven", 8:"Eight", 9:"Nine"}

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