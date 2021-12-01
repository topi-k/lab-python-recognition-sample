# -*- coding: utf-8 -*-
# データセット用既存画像変換用プログラム
# 2021/11/24

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
# Settings
margin = 10
save_img_cnt = 0

while(1):
    user_id = input("UserID:")
    if user_id.isdecimal():
        if os.path.exists("./images/right/"+user_id) == False:
            os.mkdir("./images/right/"+user_id)
        if os.path.exists("./images/left/"+user_id) == False:
            os.mkdir("./images/left/"+user_id)
        user_id = int(user_id)
        break

if os.path.exists(sys.argv[1]) == False or sys.argv[1] == "":
    print("ファイルが存在しません。 File:",sys.argv[1])
    sys.exit()

raw_image = cv2.imread(sys.argv[1])

gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

ear_right = ear_right_cascade.detectMultiScale(gray)
ear_left = ear_left_cascade.detectMultiScale(gray)

time_unique = int(time.time())

for (ercx, ercy, ercw, erch) in ear_right:
    #cv2.rectangle(img, (ercx, ercy),(ercx+ercw, ercy+erch), (0, 0, 255), 2)

    save_img_1 = raw_image[ercy-margin:ercy+erch +
        margin, ercx-margin:ercx+ercw+margin]
    cv2.imwrite(
        "./images/right/{0}/save_img_right_{1}.jpg".format(user_id, time_unique), save_img_1)
    print(
        "./images/right/{0}/save_img_right_{1}.jpg".format(user_id, time_unique))
    
    save_img_cnt = save_img_cnt + 1
    cv2.namedWindow("save_img", cv2.WINDOW_AUTOSIZE)
    cv2.imshow('save_img', save_img_1)

for (elcx, elcy, elcw, elch) in ear_left:
    #cv2.rectangle(img, (elcx, elcy),(elcx+elcw, elcy+elch), (0, 0, 255), 2)
    save_img_2 = raw_image[elcy-margin:elcy+elch +
         margin, elcx-margin:elcx+elcw+margin]
    cv2.imwrite(
        "./images/left/{0}/save_img_left_{1}.jpg".format(user_id, time_unique), save_img_2)
    print(
        "./images/left/{0}/save_img_left_{1}.jpg".format(user_id, time_unique))
    
    save_img_cnt = save_img_cnt + 1
    cv2.namedWindow("save_img", cv2.WINDOW_AUTOSIZE)
    cv2.imshow('save_img', save_img_2)

if save_img_cnt != 0:
    print(save_img_cnt,"個の耳が検知され、正常に保存されました。")
else:
    print("写真から耳を検出できませんでいた。")

cv2.waitKey(0)
cv2.destroyAllWindows()
