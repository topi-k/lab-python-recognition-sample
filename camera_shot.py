# -*- coding: utf-8 -*-
# Wiki https://en.wikipedia.org/wiki/Haar-like_feature
# OpenCV-Python　https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html　

import cv2
import sys
import time
import os

# Haar-like特徴分類器の読み込み
face_cascade = cv2.CascadeClassifier(
    'data/haarcascades/haarcascade_frontalface_default.xml')

ear_right_cascade = cv2.CascadeClassifier(
    'data/haarcascades/haarcascade_mcs_rightear.xml')
ear_left_cascade = cv2.CascadeClassifier(
    'data/haarcascades/haarcascade_mcs_leftear.xml')
margin = 10

# カメラから顔をキャプチャ
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Fail to open videocapture")
    sys.exit()


while(1):
    user_id = input("UserID:")
    if user_id.isdecimal():
        if os.path.exists("./images/right/"+user_id) == False:
            os.mkdir("./images/right/"+user_id)
        if os.path.exists("./images/left/"+user_id) == False:
            os.mkdir("./images/left/"+user_id)
        user_id = int(user_id)
        break

while(True):
    ret, img = cap.read()

# グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ear_right = ear_right_cascade.detectMultiScale(gray)
    ear_left = ear_left_cascade.detectMultiScale(gray)

    time_unique = int(time.time())

    for (ercx, ercy, ercw, erch) in ear_right:
        #cv2.rectangle(img, (ercx, ercy),(ercx+ercw, ercy+erch), (0, 0, 255), 2)

        save_img_1 = img[ercy-margin:ercy+erch +
                         margin, ercx-margin:ercx+ercw+margin]
        cv2.imwrite(
            "./images/right/{0}/save_img_right_{1}.jpg".format(user_id, time_unique), save_img_1)
        print(
            "./images/right/{0}/save_img_right_{1}.jpg".format(user_id, time_unique))

    for (elcx, elcy, elcw, elch) in ear_left:
        #cv2.rectangle(img, (elcx, elcy),(elcx+elcw, elcy+elch), (0, 0, 255), 2)
        save_img_2 = img[elcy-margin:elcy+elch +
                         margin, elcx-margin:elcx+elcw+margin]
        cv2.imwrite(
            "./images/left/{0}/save_img_left_{1}.jpg".format(user_id, time_unique), save_img_2)
        print(
            "./images/left/{0}/save_img_left_{1}.jpg".format(user_id, time_unique))

    cv2.imshow('img', img)
    time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
