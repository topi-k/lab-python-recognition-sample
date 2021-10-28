#OpenCVとosをインポート
import cv2
import os
 
TARGET_FILE = "ear_test_left.jpg"
IMG_DIR = 'images/'
IMG_SIZE = (500, 500)

target_img_path = IMG_DIR + TARGET_FILE
#ターゲット画像をグレースケールで読み出し
target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)

print(target_img)

#ターゲット画像を200px×200pxに変換
#target_img = cv2.resize(target_img, IMG_SIZE)

# BFMatcherオブジェクトの生成
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# AKAZEを適用、特徴点を検出
detector = cv2.AKAZE_create()
(target_kp, target_des) = detector.detectAndCompute(target_img, None)

print('TARGET_FILE: %s' % (TARGET_FILE))

files = os.listdir(IMG_DIR)
for file in files:
    if file == '.DS_Store' or file == TARGET_FILE:
        continue
    #比較対象の写真の特徴点を検出
    comparing_img_path = IMG_DIR + file
    try:
        comparing_img = cv2.imread(comparing_img_path, cv2.IMREAD_GRAYSCALE)
        #comparing_img = cv2.resize(comparing_img, IMG_SIZE)
        cv2.namedWindow("comparing_img", cv2.WINDOW_NORMAL)
        cv2.imshow('comparing_img', comparing_img)

        print(comparing_img.shape,type(comparing_img))
        print(target_img.shape,type(target_img))

        (comparing_kp, comparing_des) = detector.detectAndCompute(comparing_img, None)
        # BFMatcherで総当たりマッチングを行う

        print(type(target_des))
        print(type(comparing_des))

        matches = bf.match(target_des, comparing_des)
        
        #特徴量の距離を出し、平均を取る
        dist = [m.distance for m in matches]
        ret = sum(dist) / len(dist)
    except cv2.error as error:
        print(error)
        # cv2がエラーを吐いた場合の処理
        ret = -1

    print(file, ret)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()