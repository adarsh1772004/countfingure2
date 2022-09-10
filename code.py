import mediapipe as mp
import cv2
camera = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8,
                       min_tracking_confidence=0.5)
tipid = [8, 12, 16, 20]


def countfingures(img, hand_landmarks, handnumber=0):
    if hand_landmarks:
        landmarks = hand_landmarks[handnumber].landmark
        fingers=[]
        for lm in tipid:
            fingerTipy=landmarks[lm].y
            fingerBottomy=landmarks[lm-2].y
            if fingerTipy<fingerBottomy:
                fingers.append(1)
            if fingerTipy>fingerBottomy:
                fingers.append(0)
        totalfinger=fingers.count(1)
        text=f'Fingers: {totalfinger}'
        cv2.putText(img,text,(100, 50),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)



def drawhandlandmarks(img, hand_landmarks):
    if (hand_landmarks):
        for l in hand_landmarks:
            mp_drawing.draw_landmarks(img, l, mp_hands.HAND_CONNECTIONS)


while True:
    success, img = camera.read()
    img = cv2.flip(img, 1)
    results = hands.process(img)
    hand_landmarks = results.multi_hand_landmarks
    drawhandlandmarks(img, hand_landmarks)
    countfingures(img, hand_landmarks)
    cv2.imshow("reasults", img)
    if cv2.waitKey(1) == 32:
        break
cv2.destroyAllWindows()
