import cv2
import pyautogui

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

WykrywanieTwarzy = cv2.CascadeClassifier('Haar_classcades/haarcascade_frontalface_default.xml')
WykrywanieOczu = cv2.CascadeClassifier('Haar_classcades/haarcascade_eye.xml')

while True:
    ret, frame = capture.read()
    if not ret:
        break

    skalaSzarosci = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    innykolor = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS_FULL)

    facescascade = WykrywanieTwarzy.detectMultiScale(
        skalaSzarosci,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(70, 70)
    )

    for x, y, face_width, face_height in facescascade:
        cv2.rectangle(frame, (x, y), (x + face_width, y + face_height), (255, 0, 0), 5)
        face_roi = frame[y:y + face_height, x:x + face_width]
        blur = cv2.blur(face_roi, (15, 15))
        frame[y:y + face_height, x:x + face_width] = blur

    eyescascade = WykrywanieOczu.detectMultiScale(
        innykolor,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(25, 25)
    )

    for x, y, eyes_width, eyes_height in eyescascade:
        cv2.rectangle(frame, (x, y), (x + eyes_width, y + eyes_width), (100, 200, 46), 2)

    cv2.imshow('kolor', frame)
    cv2.imshow('szarość', skalaSzarosci)
    cv2.imshow('innykolor', innykolor)

    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()