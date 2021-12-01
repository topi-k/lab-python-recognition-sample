# -*- coding: utf-8 -*-
# AKAZE(OpenCV) 動作検証用コード
# 2021/10/23

#OpenCVをインポート
import cv2

# １枚目の画像を読み込む
img1 = cv2.imread("./images/ear_test_left.jpg")  
# ２枚目の画像を読み込む
img2 = cv2.imread("./images/save_img_left_1632986435.jpg") 

img1 = cv2.resize(img1, dsize=None, fx=3, fy=3)
img2 = cv2.resize(img2, dsize=None, fx=3, fy=3)

print("size:",img1.shape,img2.shape)

# １枚目の画像をグレースケールで読み出し
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY) 
# ２枚目の画像をグレースケールcd で読み出し
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY) 

# コントラスト均等化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray1 = clahe.apply(gray1)
gray2 = clahe.apply(gray2)

# AKAZE検出器の生成
akaze = cv2.AKAZE_create() 
# gray1にAKAZEを適用、特徴点を検出
kp1, des1 = akaze.detectAndCompute(gray1,None) 
# gray2にAKAZEを適用、特徴点を検出
kp2, des2 = akaze.detectAndCompute(gray2,None) 

cv2.namedWindow("Gray1", cv2.WINDOW_NORMAL)
cv2.imshow('Gray1', gray1)
cv2.namedWindow("Gray2", cv2.WINDOW_NORMAL)
cv2.imshow('Gray2', gray2)

print("size:",img1.shape,img2.shape)
print("size:",gray1.shape,gray2.shape)
print("kp:",type(kp1),type(kp2))
print("des:",type(des1),type(des2))

cv2.waitKey(0)

# BFMatcherオブジェク
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptorsを生成
matches = bf.match(des1, des2)

# matchesをdescriptorsの似ている順にソートする 
matches = sorted(matches, key = lambda x:x.distance)

# 検出結果を描画する
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#検出結果を描画した画像を出力する
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.imshow('Result', img3)

cv2.waitKey(0)
cv2.destroyAllWindows()