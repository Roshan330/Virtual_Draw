import cv2
import mediapipe as mp
import os
import numpy as np

# Load header images
folderPath = "header1"
myList = os.listdir(folderPath)
overlayList = [cv2.imread(f'{folderPath}/{imPath}') for imPath in myList]
header1 = overlayList[0]  # Default header image

# Video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# MediaPipe hand detector
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Drawing variables
drawColor = (255, 0, 255)  # Default pink
brushThickness = 15
eraserThickness = 70
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
smoothening = 0.7

# Header area height
headerHeight = 118

# Color selection regions 🖼
colorRegions = {
    "pink": (100, 300),
    "red": (300, 400),
    "blue": (400, 500),
    "green": (500, 600),
    "grey": (600,700),
    "orange": (700,800),
    "black": (900, 1000),
    "eraser": (1050, 1280)
}

# Initial smoothed coordinates
smooth_x, smooth_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(imgRGB)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if len(lmList) != 0:
                # Detect fingers up
                fingers = []
                if lmList[4][1] > lmList[3][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                for tip in [8, 12, 16, 20]:
                    if lmList[tip][2] < lmList[tip - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                ix, iy = lmList[8][1], lmList[8][2]

                # Smooth the index finger point
                smooth_x = int(smoothening * smooth_x + (1 - smoothening) * ix)
                smooth_y = int(smoothening * smooth_y + (1 - smoothening) * iy)

                # Show only the index fingertip dot in drawColor
                cv2.circle(frame, (smooth_x, smooth_y), 10, drawColor, cv2.FILLED)

                # Selection mode: Two fingers up
                if fingers[1] == 1 and fingers[2] == 1:
                    xp, yp = 0, 0
                    if iy < headerHeight:
                        for color, (x1, x2) in colorRegions.items():
                            if x1 < ix < x2:
                                if color == "eraser":
                                    drawColor = (0, 0, 0)
                                elif color == "pink":
                                    drawColor = (255, 0, 255)
                                elif color == "red":
                                    drawColor = (0, 0, 255)
                                elif color == "blue":
                                    drawColor = (255, 165, 0)
                                elif color == "green":
                                    drawColor = (75, 205, 45)
                                elif color == "grey":
                                    drawColor = (128, 128, 128)
                                elif color == "orange":
                                    drawColor = (0, 165, 255)
                                elif color == "black":
                                    drawColor = (255, 255, 255)
                                break

                # Drawing mode: Only index finger up
                if fingers[1] == 1 and fingers[2] == 0:
                    if iy > headerHeight:
                        if xp == 0 and yp == 0:
                            xp, yp = smooth_x, smooth_y
                        if drawColor == (0, 0, 0):
                            cv2.line(imgCanvas, (xp, yp), (smooth_x, smooth_y), drawColor, eraserThickness)
                        else:
                            cv2.line(imgCanvas, (xp, yp), (smooth_x, smooth_y), drawColor, brushThickness)
                        xp, yp = smooth_x, smooth_y

    # Combine drawing and webcam frame
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, imgInv)
    frame = cv2.bitwise_or(frame, imgCanvas)

    # Show header
    header1 = cv2.resize(overlayList[0], (1280, headerHeight))
    frame[0:headerHeight, 0:1280] = header1

    # Show the output
    cv2.imshow("Virtual Drawing", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
