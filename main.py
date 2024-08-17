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
            img_surface = DISPLAY_SURFACE.subsurface(pygame.Rect(rect_min_x, rect_min_y, rect_max_x-rect_min_x, rect_max_y-rect_min_y)) 
            img_arr = pygame.surfarray.array3d(img_surface) 
            img_arr = np.transpose(img_arr, (1, 0, 2))  
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

            # predict the number from handwritten digit!
            if PREDICT: 

                # resize user drawn number, add padding, normalize
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, ((10, 10), (10, 10)), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28)) / 255.0

                # add batch/channel dims, label image with prediction
                image = image.reshape(1, 28, 28, 1)  
                label = str(LABELS[np.argmax(MODEL.predict(image))])
                
                # render the text and rectangle object
                text = FONT.render(label, True, RED, WHITE)
                rect_obj = pygame.Rect(rect_min_x, rect_min_y, rect_max_x-rect_min_x, rect_max_y-rect_min_y) 
                text_rect = text.get_rect(center=(rect_min_x + (rect_max_x - rect_min_x) // 2, rect_max_y + 20))

                # draw text and rectangle onto display surface
                pygame.draw.rect(DISPLAY_SURFACE, RED, rect_obj, 2) 
                DISPLAY_SURFACE.blit(text, text_rect) 

    pygame.display.update()