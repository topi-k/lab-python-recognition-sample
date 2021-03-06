# -*- coding: utf-8 -*-
#　カメラから入力された画像を登録画像として登録するためのプログラム
# 2021/11/03

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
margin = 0

right_ear_img_cnt = 0
left_ear_img_cnt = 0

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

    ear_right = ear_right_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(50,50))
    ear_left = ear_left_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(50,50))

    time_unique = int(time.time())

    for (ercx, ercy, ercw, erch) in ear_right:
        #cv2.rectangle(img, (ercx, ercy),(ercx+ercw, ercy+erch), (0, 0, 255), 2)

        save_img_1 = img[ercy-margin:ercy+erch +
                         margin, ercx-margin:ercx+ercw+margin]
        if sys.argv[1] == "raw":
            cv2.imwrite(
                "./tests/images/raw/save_img_right_{1}_{0}.jpg".format(user_id, time_unique), img)
            print(
             "./tests/images/raw/save_img_right_{1}_{0}.jpg".format(user_id, time_unique))
             
        else:
            cv2.imwrite(
                "./images/right/{0}/save_img_right_{1}.jpg".format(user_id, time_unique), save_img_1)
            print(
                "./images/right/{0}/save_img_right_{1}.jpg".format(user_id, time_unique))

        right_ear_img_cnt = right_ear_img_cnt + 1

        cv2.namedWindow("save_img", cv2.WINDOW_AUTOSIZE)
        cv2.imshow('save_img', save_img_1)

    for (elcx, elcy, elcw, elch) in ear_left:
        #cv2.rectangle(img, (elcx, elcy),(elcx+elcw, elcy+elch), (0, 0, 255), 2)
        save_img_2 = img[elcy-margin:elcy+elch +
                         margin, elcx-margin:elcx+elcw+margin]
        if sys.argv[1] == "raw":
            cv2.imwrite(
                "./tests/images/raw/save_img_left_{1}_{0}.jpg".format(user_id, time_unique), img)
            print(
                "./tests/images/raw/save_img_left_{1}_{0}.jpg".format(user_id, time_unique))
        else:
            cv2.imwrite(
                "./images/left/{0}/save_img_left_{1}.jpg".format(user_id, time_unique), save_img_2)
            print(
                "./images/left/{0}/save_img_left_{1}.jpg".format(user_id, time_unique))           

        left_ear_img_cnt = left_ear_img_cnt + 1
        
        cv2.namedWindow("save_img", cv2.WINDOW_AUTOSIZE)
        cv2.imshow('save_img', save_img_2)


    cv2.imshow('img', img)
    time.sleep(1)
    print(left_ear_img_cnt, "/10:", right_ear_img_cnt, "/10")

    if left_ear_img_cnt > 10 and right_ear_img_cnt > 10:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
