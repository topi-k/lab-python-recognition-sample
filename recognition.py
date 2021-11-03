# -*- coding: utf-8 -*-
# Wiki https://en.wikipedia.org/wiki/Haar-like_feature
# OpenCV-Python　https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html　

import cv2
import sys
import time
import os
import numpy as np

device = cv2.ocl.Device_getDefault()
print(f"Vendor ID: {device.vendorID()}")
print(f"Vendor name: {device.vendorName()}")
print(f"Name: {device.name()}")

# Haar-like特徴分類器の読み込み
ear_right_cascade = cv2.CascadeClassifier(
    'data/haarcascades/haarcascade_mcs_rightear.xml')
ear_left_cascade = cv2.CascadeClassifier(
    'data/haarcascades/haarcascade_mcs_leftear.xml')
face_cascade = cv2.CascadeClassifier(
    'data/haarcascades/haarcascade_frontalface_default.xml')

# A-KAZE/KNN Setting
margin = 10 # 画像の周辺空間
zoom_diam = 3 # （特徴量検出用の拡大倍率）
threshold = 200 # 閾値

def main():

    # カメラから顔をキャプチャ
    cap = cv2.VideoCapture(0)

    recog_user = -1
    distance = 0

    if not cap.isOpened():
        print("Fail to open videocapture")
        sys.exit()

    while(True):
        time_start = time.time()
        ret, img = cap.read()

        # グレースケール変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 顔・耳のパーツを検知する
        ear_right = ear_right_cascade.detectMultiScale(gray)
        ear_left = ear_left_cascade.detectMultiScale(gray)
        faces = face_cascade.detectMultiScale(gray)

        # 顔を検知していたら誤検知防止の為に処理を切り上げる
        if len(faces) > 0:
            continue

        # 右耳の検知処理
        for (ercx, ercy, ercw, erch) in ear_right:
            IMG_DIR = os.path.abspath(
                os.path.dirname(__file__)) + '/images/right/'
            right_ear_img = img[ercy-margin:ercy+erch+margin, ercx-margin:ercx+ercw+margin]
            recog_user, distance = recognition(right_ear_img, IMG_DIR)
            time_end = time.time()

        # 左耳の検知処理
        for (elcx, elcy, elcw, elch) in ear_left:
            IMG_DIR = os.path.abspath(
                os.path.dirname(__file__)) + '/images/left/'
            left_ear_img = img[elcy-margin:elcy+elch+margin, elcx-margin:elcx+elcw+margin]
            recog_user, distance = recognition(left_ear_img, IMG_DIR)
            time_end = time.time()
        
        if recog_user != -1:
            print ("TIME: {0}".format((time_end - time_start) * 1000 / 10000) + "[sec]")
            print ("Recognition >> User:",recog_user," Ret",distance)
            break

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def recognition(img, IMG_DIR):
    file_dir = ""
    users_dir = os.listdir(IMG_DIR)
    users_cnt = len(users_dir)

    # AKAZE検出器の生成
    akaze = cv2.AKAZE_create()
    # BFMatcherオブジェクトの生成
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # CLAHEオブジェクトの生成
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))

    # カメラの画像を処理する
    camera_img = img

    # 画像を指定した倍率に拡大する
    camera_img = cv2.resize(camera_img, dsize=None, fx=zoom_diam, fy=zoom_diam)
    # グレースケール変換
    camera_img = cv2.cvtColor(camera_img,cv2.COLOR_BGR2GRAY)
    # コントラスト均等化
    camera_img = clahe.apply(camera_img)

    (target_kp, target_des) = akaze.detectAndCompute(camera_img, None)

    user_lst_ret = list(range(users_cnt))
    user_lst_id = list(range(users_cnt))

    index = 0

    for user_dir in users_dir:
        
        temp = 0

        user_id = int(user_dir)
        user_lst_id[index] = user_id

        files_dir = os.listdir(IMG_DIR+"/" + user_dir + "/")

        for file_dir in files_dir:
            lib_img_path = IMG_DIR + "/" + user_dir + "/" + file_dir # ライブラリの検出対象画像へのパス

            # ライブラリに登録されている画像をロードして特徴量検出を行う
            # ロードしたライブラリの画像を処理する
            lib_img = cv2.imread(lib_img_path)
            # 画像を指定した倍率に拡大する
            lib_img = cv2.resize(lib_img, dsize=None, fx=zoom_diam, fy=zoom_diam)
            # グレースケール変換
            lib_img = cv2.cvtColor(lib_img, cv2.COLOR_BGR2GRAY)
            # コントラスト均等化
            lib_img = clahe.apply(lib_img)

            (comparing_kp, comparing_des) = akaze.detectAndCompute(lib_img, None)

            # BFMatcher で ライブラリに登録されている全ての画像に対してマッチングを行う
            matches = bf.match(target_des, comparing_des)
            if len(matches) == 0:
                break
            
            dist = [m.distance for m in matches]
            ret = sum(dist) / len(dist)
            temp = ret + temp
        
        user_lst_ret[index] = temp / len(files_dir)
        user_lst_ret[index] = np.amax(user_lst_ret[index])

        index = index + 1

    recognition_user = -1
    recognition_ret = 0

    for i in range(index):
        if recognition_ret < user_lst_ret[i]:
            recognition_ret = user_lst_ret[i]
            recognition_user = user_lst_id[i]
        print("User:", user_lst_id[i] ," Ret:", user_lst_ret[i])
    
    return recognition_user,recognition_ret



if __name__ == "__main__":
    main()
