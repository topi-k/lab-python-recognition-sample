import cv2
import sys
import time
import os
import numpy as np

zoom_diam = 5 # （特徴量検出用の拡大倍率）

img = cv2.imread(sys.argv[1]) 
img = cv2.resize(img, dsize=None, fx=zoom_diam, fy=zoom_diam)

# AKAZE検出器の生成
akaze = cv2.AKAZE_create()
# CLAHEオブジェクトの生成
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))
# カメラの画像を処理する
base_img = img

# グレースケール変換
gray_img = cv2.cvtColor(base_img,cv2.COLOR_BGR2GRAY)
# コントラスト均等化
clahe_img = clahe.apply(gray_img)

cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
cv2.imshow('img', img)

cv2.namedWindow("gray_img", cv2.WINDOW_AUTOSIZE)
cv2.imshow('gray_img', gray_img)
gray_kp, gray_des = akaze.detectAndCompute(gray_img, None)
gray_kp_img = cv2.drawKeypoints(gray_img, gray_kp, None);
cv2.namedWindow("gray_kp_img", cv2.WINDOW_AUTOSIZE)
cv2.imshow('gray_kp_img', gray_kp_img)

cv2.namedWindow("clahe_img", cv2.WINDOW_AUTOSIZE)
cv2.imshow('clahe_img', clahe_img)
clahe_kp, clahe_des = akaze.detectAndCompute(clahe_img, None)
clahe_kp_img = cv2.drawKeypoints(clahe_img, clahe_kp, None);
cv2.namedWindow("clahe_kp_img", cv2.WINDOW_AUTOSIZE)
cv2.imshow('clahe_kp_img', clahe_kp_img)

cv2.waitKey(0)
cv2.destroyAllWindows()