'''
code used to create training data, heavily uses the boilerplate code from Google's Mediapipe website
https://google.github.io/mediapipe/solutions/hands

The program waits for a user to enter 1 on the terminal, 
This time is used for the user to hold the other hand in front of the camera that holds a particular sign language alphabet
Then the program uses the camera of the device to record 
  100 frames of the user's hands, and extracts the coordinates of 21 different points of the hand.
One can slighly move the hand (that shows the alphabet) slightly around, and move it towards & away from the screen to create 'better' training data
The coordinates are 3D coordinates in meters, from what is roughly the geometric center of the hand (from: https://google.github.io/mediapipe/solutions/hands)
This serves as the training sample of a particular alphabet
'''

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import numpy as np

# the number of instances recorded
storeSize_ = 100
# for each frame, we have 21 points on the hand recorded, each point being a 3D coordinate
store = np.zeros((storeSize_, 21, 3))

startFlag = False
while not startFlag:
  inp = input("Hit 1 to start recording samples")
  startFlag = True if inp == '1' else False

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    max_num_hands=1,
    min_tracking_confidence=0.5) as hands:
  # while cap.isOpened():
  for counter in range(storeSize_):
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
        for i in range(21):
          store[counter, i, 0] = hand_landmarks.landmark[i].x
          store[counter, i, 1] = hand_landmarks.landmark[i].y
          store[counter, i, 2] = hand_landmarks.landmark[i].z
        print(counter)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
print("finished recording samples")
# set file name here, we can only store 2D shaped data in text file, so 21*3 -> 63*1
np.savetxt('train_data/c_val.txt', store.reshape(storeSize_, 63))
print("saved samples")
