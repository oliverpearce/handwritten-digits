import pygame, sys
from pygame.locals import *
import numpy as np
from keras.api.models import load_model
import cv2

# constants
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
BOUNDARY_WIDTH = 5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

MODEL = load_model("bestmodel.keras")
LABELS = {0:"Zero", 1:"One", 2:"Two", 3:"Three", 4:"Four", 5:"Five", 6:"Six", 7:"Seven", 8:"Eight", 9:"Nine"}

# initialize pygame window
pygame.init()

# configure pygame display
DISPLAY_SURFACE = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("window of doom")

# init variables
FONT = pygame.font.Font("freesansbold.ttf", 18)
writing = False
xcord_list = []
ycord_list = []
image_count = 1
PREDICT = True

# forever loop for display processing
while 1:
    for event in pygame.event.get():
        # quit the menu
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        # write to window
        if event.type == MOUSEMOTION and writing:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAY_SURFACE, WHITE, (xcord, ycord), 4, 0)

            xcord_list.append(xcord)
            ycord_list.append(ycord)

         # write if mouse clicked down
        if event.type == MOUSEBUTTONDOWN:
            writing = True

        # stop writing when mouse is let go
        if event.type == MOUSEBUTTONUP:
            writing = False

            # sort coordinate list
            xcord_list = sorted(xcord_list)
            ycord_list = sorted(ycord_list)

            # get bounds for rectangle
            rect_min_x = max(xcord_list[0] - BOUNDARY_WIDTH, 0)
            rect_max_x = min(WINDOW_WIDTH, xcord_list[-1] + BOUNDARY_WIDTH)
            rect_min_y = max(ycord_list[0] - BOUNDARY_WIDTH, 0)
            rect_max_y = min(WINDOW_HEIGHT, ycord_list[-1] + BOUNDARY_WIDTH)

            # reset coord list
            xcord_list = []
            ycord_list = []

            # create image array to pass to model!
            img_arr = np.array(pygame.PixelArray(DISPLAY_SURFACE))[rect_min_x:rect_max_x, rect_min_y:rect_max_y]
            img_arr = img_arr.T.astype(np.float32)

            if PREDICT: 
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28)) / 255

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])

                text = FONT.render(label, True, RED, WHITE)
                # pygame.draw.rect(DISPLAY_SURFACE, RED, pygame.Rect(30, 30, 60, 60),  2)
                rect_obj = pygame.Rect(rect_min_x, rect_min_y, rect_max_x, rect_max_y)
                # rect_obj.left, rect_obj.bottom = rect_min_x, rect_min_y

                DISPLAY_SURFACE.blit(text, rect_obj)

    pygame.display.update()