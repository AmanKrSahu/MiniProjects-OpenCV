import cv2 as cv
import mediapipe as mp
import time

video = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


pTime = 0
cTime = 0

while True:
  isTrue, img = video.read()

  # Converting into RGB format
  imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

  # Extracting the result
  results = hands.process(imgRGB)

  # Checking if the hands are detected
  if results.multi_hand_landmarks:
    # To extract information about each hand
    for handLms in results.multi_hand_landmarks:
      # To extract information about each landmark
      for id, lm in enumerate(handLms.landmark):
        h, w, c = img.shape
        # Converting decimal values to pixel value
        cx, cy = int(lm.x*w), int(lm.y*h)
        # Printing pixel value of each landmark along with its id
        print(id, cx, cy)
      # To draw points & connections on the hands
      mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

  # Calculating fps
  cTime = time.time()
  fps = 1/(cTime-pTime)
  pTime = cTime

  # Putting fps on the screen
  cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

  cv.imshow("Video",img)
  if cv.waitKey(20) & 0xFF == ord('q'):
    break

video.release()
cv.destroyAllWindows()