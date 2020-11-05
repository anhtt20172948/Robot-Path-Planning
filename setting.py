import numpy as np
WIDTH = 1000
HEIGHT = 500
IMAGE_SIZE = [WIDTH, HEIGHT]
BLANK_IMAGE = np.zeros((HEIGHT, WIDTH , 3), np.uint8)

START_POINT = (int(WIDTH/10), int(HEIGHT - HEIGHT/10))
END_POINT = (int(WIDTH - WIDTH/10), int(HEIGHT/10))

ROI = [WIDTH/20, HEIGHT/20, WIDTH*(1-1/20), HEIGHT*(1-1/20)]
Polygon_ROI = [(ROI[0], ROI[1]), (ROI[0], ROI[3]), (ROI[2], ROI[3]), (ROI[2], ROI[1])]
ROI = [int(_) for _ in ROI]

# COLOR:
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
RED = (0, 0, 255)

FONT_SCALE = 0.7
