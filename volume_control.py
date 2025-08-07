import cv2
import numpy as np
import mediapipe as mp
import math

# Windows volume control
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Get audio device and interface
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get min and max volume range
min_vol, max_vol, _ = volume.GetVolumeRange()

# 🔊 Set default volume to 50%
default_vol = np.interp(50, [0, 100], [min_vol, max_vol])
volume.SetMasterVolumeLevel(default_vol, None)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            h, w, _ = img.shape
            for lm in handLms.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

            if len(lmList) >= 9:
                # Thumb and index tip
                x1, y1 = lmList[4]
                x2, y2 = lmList[8]
                length = math.hypot(x2 - x1, y2 - y1)

                # Draw connection and points
                cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Map gesture to volume
                vol = np.interp(length, [30, 150], [min_vol, max_vol])
                vol_percent = np.interp(length, [30, 150], [0, 100])
                volume.SetMasterVolumeLevel(vol, None)

                # Show volume percent
                cv2.putText(img, f'Volume: {int(vol_percent)}%', (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Live Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
