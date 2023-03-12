import cv2 as cv
import mediapipe as mp
import time

class handDetector():
   def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
      self.mode = mode
      self.maxHands = maxHands
      self.detectionCon = detectionCon
      self.trackCon = trackCon

      self.mpHands = mp.solutions.hands
      self.hands = self.mpHands.Hands(static_image_mode = self.mode, max_num_hands = self.maxHands, 
                                      min_detection_confidence = self.detectionCon, min_tracking_confidence = self.trackCon)
      self.mpDraw = mp.solutions.drawing_utils
    
   def findHands(self, img, draw=True):
      imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
      
      self.results = self.hands.process(imgRGB)

      if self.results.multi_hand_landmarks:
         for handLms in self.results.multi_hand_landmarks:
            if draw:
               self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
      
      return img
   
   def findPosition(self, img, handNo=0, draw=True):
      lmList = []

      if self.results.multi_hand_landmarks:
        myHand = self.results.multi_hand_landmarks[handNo]
        
        for id, lm in enumerate(myHand.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            lmList.append([id, cx, cy])

      return lmList
      
def main():
   video = cv.VideoCapture(0)
   
   pTime = 0
   cTime = 0
   
   detector = handDetector()

   while True:
    isTrue, img = video.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
       print(lmList[0])

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, 'fps:' + str(int(fps)), (10,30), cv.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)

    cv.imshow("Video",img)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break
    
   video.release()
   cv.destroyAllWindows()

if __name__ == '__main__':
   main()