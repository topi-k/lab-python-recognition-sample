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
ratio = 0.8  # A-KAZE/KNN Ratio(~1)
good_ratio = 1  # Good Ratio(0~2)

expand_template = 2 # 拡大率
margin = 30

def main():

    # カメラから顔をキャプチャ
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Fail to open videocapture")
        sys.exit()

    while(True):
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

        # 左耳の検知処理
        for (elcx, elcy, elcw, elch) in ear_left:
            IMG_DIR = os.path.abspath(
                os.path.dirname(__file__)) + '/images/left/'
            left_ear_img = img[elcy-margin:elcy+elch+margin, elcx-margin:elcx+elcw+margin]
            recog_user, distance = recognition(left_ear_img, IMG_DIR)

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def recognition(img, IMG_DIR):
    file_dir = ""
    users_dir = os.listdir(IMG_DIR)

    # AKAZE検出器の生成
    akaze = cv2.AKAZE_create()

    for user_dir in users_dir:
        time_start = time.time()
        files_dir = os.listdir(IMG_DIR+"/" + user_dir + "/")

        for file_dir in files_dir:
            lib_img_path = IMG_DIR + "/" + user_dir + "/" + file_dir

            lib_img = cv2.imread(lib_img_path)
            camera_img = img

            # 比較画像を読み込み、各種処理を実行する
            #  (glay_img_ref = 照合先画像 | camera_img = カメラ映像)
            gpu = cv2.cuda_GpuMat() # Allocate device memory only once, as memory allocation seems to take time...

            # ライブラリ側の処理
            gpu.upload(lib_img) # Upload to GPU memory
            lib_img = cv2.cvtColor(lib_img, cv2.COLOR_BGR2GRAY)
            height, width = lib_img.shape[:2]
            lib_img = cv2.resize(lib_img, None, fx = expand_template, fy = expand_template)        
            lib_img = gpu.download()

            lib_img = cv2.UMat(lib_img)   

            # カメラ画像の処理
            gpu.upload(camera_img) # Upload to GPU memory
            gpu = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2GRAY)
            height, width = camera_img.shape[:2]
            gpu = cv2.cuda.resize(gpu, (height * expand_template, width * expand_template))
            camera_img = gpu.download()

            camera_img = cv2.UMat(camera_img)

            # 特徴量を計算する
            kp1, des1 = akaze.detectAndCompute(lib_img, None)
            kp2, des2 = akaze.detectAndCompute(camera_img, None)
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)

            # データを間引きする
            good = []
            for m, n in matches:
                if m.distance < ratio * n.distance:
                    good.append(m)

            if len(good) > good_ratio:
                print("Detect user", user_dir, ". mdist:",
                    m.distance, " ndist:", ratio * n.distance, " good:", len(good))
                # 対応する特徴点同士を描画
                img_result = cv2.drawMatches(
                    lib_img, kp1, camera_img, kp2, good, None, flags=2)
                # 画像表示
                cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
                cv2.imshow('Result', img_result)
        time_end = time.time()
        print ("GPU = {0}".format((time_end - time_start) * 1000 / 10000) + "[msec]")

    return "none", 0


if __name__ == "__main__":
    main()
