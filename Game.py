# Import
import os
import random
import pygame
import SceneManager
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from Fruit import Fruit
import pymunk
import time
from Button import ButtonImg


def Game():
    # Initialize
    pygame.init()
    pygame.event.clear()

    # Create Window/Display
    width, height = 1280, 720
    window = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Fruit Slicer")

    # Initialize Clock for FPS
    fps = 30
    clock = pygame.time.Clock()

    # Webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, width)  # width
    cap.set(4, height)  # height

    # Images
    imgBackground = pygame.image.load("Project - Fruit Ninja/BackgroundGame.png").convert_alpha()
    imgBackground.set_alpha(225)
    imgGameOver = pygame.image.load("Project - Fruit Ninja/BackgroundGameOver.png").convert()

    # Hand Detector
    detector = HandDetector(maxHands=1, detectionCon=0.8)

    # Physics
    space = pymunk.Space()
    space.gravity = 0.0, -1000.0

    # Parameters
    timeTotal = 60

    # Variables
    fruitList = []
    timeGenerator = time.time()
    timeStart = time.time()
    index = 0, 0
    gameOver = False
    score = 0

    # Buttons
    buttonMenu = ButtonImg((800, 290), "Project - Fruit Ninja/ButtonMenu.png",
                           pathSoundClick="Sounds/click.mp3",
                           pathSoundHover="Sounds/hover.mp3")

    # Sounds
    pygame.mixer.init()
    pygame.mixer.music.load('Sounds/timer.mp3')
    pygame.mixer.Sound('Sounds/explosion.wav').play()
    pygame.mixer.music.play()

    # Fruit Path List
    pathFruitFolder = "Project - Fruit Ninja/Fruits"
    pathListFruit = os.listdir(pathFruitFolder)

    # Fruit
    def generateFruit():
        randomScale = round(random.uniform(0.6, 0.8), 2)
        randomFruitPath = pathListFruit[random.randint(0, len(pathListFruit) - 1)]
        if "bomb" in randomFruitPath:
            pathSoundSlice = 'Sounds/explosion.wav'
        else:
            pathSoundSlice = 'Sounds/slice.wav'

        fruitList.append(Fruit(space, path=os.path.join(pathFruitFolder, randomFruitPath),
                               grid=(4, 4), animationFrames=14, scale=randomScale,
                               pathSoundSlice=pathSoundSlice))

    # Main loop
    start = True
    while start:
        # Get Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                start = False
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    SceneManager.OpenScene("Menu")

            # Apply Logic
            # OpenCV
        if gameOver is False:
            success, img = cap.read()
            img = cv2.flip(img, 1)

            hands = detector.findHands(img, draw=False)

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgRGB = np.rot90(imgRGB)
            frame = pygame.surfarray.make_surface(imgRGB).convert()
            frame = pygame.transform.flip(frame, True, False)
            window.blit(frame, (0, 0))
            window.blit(imgBackground, (0, 0))

            if hands:
                hand = hands[0]
                index = hand['lmList'][8][:2]
                pygame.draw.circle(window, (0, 200, 0), index, 20)
                pygame.draw.circle(window, (200, 200, 200), index, 16)

            if time.time() - timeGenerator > 1:
                generateFruit()
                timeGenerator = time.time()

            x, y = index
            for i, fruit in enumerate(fruitList):
                if fruit:
                    fruit.draw(window)
                    checkSlice = fruit.checkSlice(x, y)
                    if checkSlice == 2:
                        gameOver = True
                        pygame.mixer.music.stop()
                    if checkSlice == 1:
                        fruitList[i] = False
                        score += 1
            index = 0, 0

            timeLeft = int(timeTotal - (time.time() - timeStart))
            if timeLeft <= 0:
                gameOver = True
                pygame.mixer.music.stop()

            font = pygame.font.Font(None, 60)
            textScore = font.render(str(score), True, (225, 225, 225))
            textTime = font.render(str(timeLeft), True, (225, 225, 225))
            window.blit(textScore, (225, 35))
            window.blit(textTime, (1100, 38))


        else:
            window.blit(imgGameOver, (0, 0))

            # Button
            buttonMenu.draw(window)
            if buttonMenu.state == "clicked":
                SceneManager.OpenScene("Menu")

            # Text Score
            font = pygame.font.Font(None, 150)
            textScore = font.render(str(score), True, (225, 225, 225))
            window.blit(textScore, (270, 450))

        # Update Display
        pygame.display.update()
        # Set FPS
        clock.tick(fps)
        space.step(1 / fps)


if __name__ == "__main__":
    Game()
