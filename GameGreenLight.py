# Import
import random
import cvzone
import pygame
import SceneManager
import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector
import time
from Button import ButtonImg


def Game():
    # Initialize
    pygame.init()
    pygame.event.clear()

    # Create Window/Display
    width, height = 1280, 720
    window = pygame.display.set_mode((width, height))
    pygame.display.set_caption("My Awesome Game")

    # Initialize Clock for FPS
    fps = 30
    clock = pygame.time.Clock()

    # Webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, width)  # width
    cap.set(4, height)  # height

    # Detector Body
    detector = PoseDetector()

    # Background Subtractor
    subtractor = cv2.createBackgroundSubtractorMOG2(3)

    # Flags
    gameStart, gameOver, gameWon = False, False, False
    greenLight, greenFirstFrame = True, True

    # Parameters
    thresholdDifficulty = 20  # higher is easier
    widthStart = 500
    widthEnd = 800
    timeTotal = 30

    # Variable
    countRed = 0
    timeStart = 0

    # Sounds
    pygame.mixer.init()
    soundShot = pygame.mixer.Sound("Sounds/shot.mp3")
    soundGreenLight = pygame.mixer.Sound("Sounds/GR-2.mp3")

    # Buttons
    buttonBack = ButtonImg((575, 475), "Buttons/ButtonBack.png", scale=0.6,
                           pathSoundClick="Sounds/click.mp3",
                           pathSoundHover="Sounds/hover.mp3")

    # Images
    imgGameWon = pygame.image.load("Project - SquidGame/Passed.png").convert()
    imgGameOver = pygame.image.load("Project - SquidGame/Eliminated.png").convert()

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
        if gameOver is False and gameWon is False:
            success, img = cap.read()

            # Find Body
            img = detector.findPose(img, draw=False)
            lmList, bboxInfo = detector.findPosition(img, draw=False)

            # Get image subtraction
            mask = subtractor.apply(img, 1)
            # cv2.imshow("Mask", mask)

            if bboxInfo:
                x, y, w, h = bboxInfo['bbox']
                # print(w)
                # If game has not started
                if gameStart is False:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 50)
                    if w < widthStart:
                        gameStart = True
                        timeStart = time.time()
                    else:
                        cvzone.putTextRect(img, "Move Back to start the Game", (350, 400))


                # After Game begins
                else:
                    # Find the time Remaining
                    timeLeft = int(timeTotal - (time.time() - timeStart))
                    cvzone.putTextRect(img, f'Time: {timeLeft}', (1050, 50))

                    # Check the number of white pixels in the body region
                    imgMaskCrop = mask[y:y + h, x:x + w]
                    ret, imgThreshold = cv2.threshold(imgMaskCrop, 20, 255, cv2.THRESH_BINARY)
                    cv2.imshow("Binary", imgThreshold)
                    whitePixels = cv2.countNonZero(imgThreshold)
                    print(whitePixels)

                    # Check if movement is present
                    if whitePixels > w * thresholdDifficulty:
                        colorMotion = (0, 0, 255)
                    else:
                        colorMotion = (0, 255, 0)

                    cv2.rectangle(img, (x, y), (x + w, y + h), colorMotion, 50)

                    # If Green Light
                    if greenLight:
                        if greenFirstFrame:  # For the first frame play sound and get start time
                            soundGreenLight.play()
                            timeStartGreen = time.time()
                            greenFirstFrame = False
                        else:  # after the first frame
                            if time.time() - timeStartGreen > 2:
                                greenLight = False
                                randomDelay = random.randint(30, 50)
                        if w > widthEnd:
                            gameWon = True
                            print("Game Won")

                    # If Red Light
                    else:
                        countRed += 1
                        if countRed > randomDelay:
                            greenLight = True
                            greenFirstFrame = True
                            countRed = 0

                        # Check for motion during red light
                        if whitePixels > w * thresholdDifficulty:
                            gameStart = False
                            gameOver = True
                            soundShot.play()

                    # Check if the time limit is up
                    if timeLeft <= 0:
                        gameOver = True
                        soundShot.play()

            # Displaying the final image
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgRGB = np.rot90(imgRGB)
            frame = pygame.surfarray.make_surface(imgRGB).convert()
            frame = pygame.transform.flip(frame, True, False)
            window.blit(frame, (0, 0))

        # Game Won
        elif gameOver is False and gameWon:
            window.blit(imgGameWon, (0, 0))
            buttonBack.draw(window)
            if buttonBack.state == 'clicked':
                SceneManager.OpenScene("Menu")

        # Game Over
        elif gameOver and gameWon is False:
            window.blit(imgGameOver, (0, 0))
            buttonBack.draw(window)
            if buttonBack.state == 'clicked':
                SceneManager.OpenScene("Menu")

        # Update Display
        pygame.display.update()
        # Set FPS
        clock.tick(fps)


if __name__ == "__main__":
    Game()
