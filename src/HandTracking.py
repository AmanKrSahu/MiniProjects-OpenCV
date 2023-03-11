import cv2 as cv
import mediapipe as mp
import time

video = cv.VideoCapture(0)

while True:
  isTrue, frame = video.read()
  cv.imshow("Video",frame)
  if cv.waitKey(20) & 0xFF == ord('q'):
    break

video.release()
cv.destroyAllWindows()