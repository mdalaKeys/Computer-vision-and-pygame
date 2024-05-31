# Import
import pygame
import cv2
import numpy as np
from Toggle import ToggleImg
from Slider import Slider
from cvzone.ColorModule import ColorFinder

# Initialize
pygame.init()

# Create Window/Display
width, height = 1280, 720
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("My Awesome Game")

# Initialize Clock for FPS
fps = 30
clock = pygame.time.Clock()

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # width
cap.set(4, 720)  # height

# Import Images
imgBackground = pygame.image.load('Background.png').convert()

# Toggles
toggleList = []
for x in range(4):
    toggleList.append(ToggleImg((283, 164 + 60 * x),
                                "Toggle/ToggleGreen.png", 0.5,
                                pathSoundClick="Sounds/click.mp3"))
    toggleList[x].state = 'on'

# Slider
sliderList = []
sliderList.append(Slider((104, 424 + 0 * 65), startValue=0, widthSlider=180, fontSize=40, max=179))
sliderList.append(Slider((104, 424 + 1 * 65), startValue=179, widthSlider=180, fontSize=40, max=179))
sliderList.append(Slider((104, 424 + 2 * 65), startValue=0, widthSlider=180, fontSize=40, max=255))
sliderList.append(Slider((104, 424 + 3 * 65), startValue=255, widthSlider=180, fontSize=40, max=255))

# Color Finder
colorFinder = ColorFinder()


def opencvToPygame(img):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgRGB = np.rot90(imgRGB)
    frame = pygame.surfarray.make_surface(imgRGB).convert()
    frame = pygame.transform.flip(frame, True, False)
    return frame


# Main loop
start = True
while start:
    # Get Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            start = False
            pygame.quit()

    # Apply Logic

    # OpenCV
    h, w, = 320, 240
    success, img = cap.read()
    img = cv2.flip(img, 1)

    #img = cv2.imread(img)

    img = cv2.resize(img, (w, h))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgCanny = cv2.Canny(imgGray, 100, 150)

    hsvVals = {'hmin': sliderList[0].value, 'smin': sliderList[2].value,
               'vmin': 0, 'hmax': sliderList[1].value, 'smax': sliderList[3].value, 'vmax': 255}
    imgColor, mask = colorFinder.update(img, hsvVals)

    # Display Opencv Images
    window.blit(imgBackground, (0, 0))
    if toggleList[0].state == 'on':
        window.blit(opencvToPygame(img), (484, 103))
    if toggleList[1].state == 'on':
        window.blit(opencvToPygame(imgGray), (865, 103))
    if toggleList[2].state == 'on':
        window.blit(opencvToPygame(imgCanny), (484, 444))
    if toggleList[3].state == 'on':
        window.blit(opencvToPygame(imgColor), (865, 444))

    for x in range(4):
        toggleList[x].draw(window)

    for slider in sliderList:
        slider.draw(window)

    # Update Display
    pygame.display.update()
    # Set FPS
    clock.tick(fps)
