# -*- coding: utf-8 -*-
# Wiki https://en.wikipedia.org/wiki/Haar-like_feature
# OpenCV-Python　https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html　

import cv2
import sys
import time
import os

# Haar-like特徴分類器の読み込み
ear_right_cascade = cv2.CascadeClassifier(
    'data/haarcascades/haarcascade_mcs_rightear.xml')
ear_left_cascade = cv2.CascadeClassifier(
    'data/haarcascades/haarcascade_mcs_leftear.xml')

# カメラから顔をキャプチャ
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Fail to open videocapture")
    sys.exit()

max_ret = 0
recognition_user = 0
threshold = 0.6

while(True):
    ret, img = cap.read()

# グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ear_right = ear_right_cascade.detectMultiScale(gray)
    ear_left = ear_left_cascade.detectMultiScale(gray)

    time_unique = int(time.time())
    for (ercx, ercy, ercw, erch) in ear_right:
        cv2.rectangle(img, (ercx, ercy),
                      (ercx+ercw, ercy+erch), (0, 0, 255), 2)

        IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '/images/right/'
        IMG_SIZE = (200, 200)

        file_dir = ""
        users_dir = os.listdir(IMG_DIR)
        max_ret = 0
        for user_dir in users_dir:
            files_dir = os.listdir(IMG_DIR+"/" + user_dir + "/")

            for file_dir in files_dir:
                target_img_path = IMG_DIR + "/" + user_dir + "/" + file_dir
                target_img = cv2.imread(target_img_path)
                target_img = cv2.resize(target_img, IMG_SIZE)
                target_hist = cv2.calcHist(
                    [target_img], [0], None, [256], [0, 256])

                comparing_img = cv2.resize(img, IMG_SIZE)
                comparing_hist = cv2.calcHist(
                    [comparing_img], [0], None, [256], [0, 256])

                ret = cv2.compareHist(target_hist, comparing_hist, 0)

                #print(user_dir, file_dir, ret)
                if (max_ret < ret and ret > threshold):
                    max_ret = ret
                    recognition_user = user_dir
                    recognition_image = file_dir
                    print("[RECOG]", recognition_user, max_ret, file_dir)

    for (elcx, elcy, elcw, elch) in ear_left:
        cv2.rectangle(img, (elcx, elcy),
                      (elcx+elcw, elcy+elch), (0, 0, 255), 2)

        IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '/images/left/'
        IMG_SIZE = (200, 200)

        file_dir = ""
        users_dir = os.listdir(IMG_DIR)
        max_ret = 0
        for user_dir in users_dir:
            files_dir = os.listdir(IMG_DIR+"/" + user_dir + "/")

            for file_dir in files_dir:
                target_img_path = IMG_DIR + "/" + user_dir + "/" + file_dir
                target_img = cv2.imread(target_img_path)
                target_img = cv2.resize(target_img, IMG_SIZE)
                target_hist = cv2.calcHist(
                    [target_img], [0], None, [256], [0, 256])

                comparing_img = cv2.resize(img, IMG_SIZE)
                comparing_hist = cv2.calcHist(
                    [comparing_img], [0], None, [256], [0, 256])

                ret = cv2.compareHist(target_hist, comparing_hist, 0)

                #print(user_dir, file_dir, ret)
                if (max_ret < ret and ret > threshold):
                    max_ret = ret
                    recognition_user = user_dir
                    recognition_image = file_dir
                    print("[RECOG]", recognition_user, max_ret, file_dir)


# 画像表示
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 何かキーを押したら終了
cv2.waitKey(0)
cv2.destroyAllWindows()
