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
zoom_diam = 2 # （特徴量検出用の拡大倍率）
threshold = 200 # 閾値

def main():
    recog_user = -1
    distance = 0

    time_start = time.time()
    img = cv2.imread(sys.argv[1]) 

    print(type(img))

    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ear_right = ear_right_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(50,50))
    ear_left = ear_left_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(50,50))

    print("Detected R:",len(ear_right)," L:",len(ear_left))
    if len(ear_right) != 0:
        for (ercx, ercy, ercw, erch) in ear_right:
            img = img[ercy-margin:ercy+erch + margin, ercx-margin:ercx+ercw+margin]
            IMG_DIR = os.path.abspath( os.path.dirname(__file__)) + '/images/right/'
            recog_user, distance = recognition(img, IMG_DIR)
            time_end = time.time()

    if len(ear_left) != 0:
        for (ercx, ercy, ercw, erch) in ear_left:
            img = img[ercy-margin:ercy+erch + margin, ercx-margin:ercx+ercw+margin]
            IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '/images/left/'
            recog_user, distance = recognition(img, IMG_DIR)
            time_end = time.time()
        
    if recog_user != -1:
        print ("TIME: {0}".format((time_end - time_start) * 1000 / 10000) + "[sec]")
        print ("Recognition >> User:",recog_user," Ret",distance)

    cv2.destroyAllWindows()


def recognition(img, IMG_DIR):
    print("Recognition processing... :",IMG_DIR)
    file_dir = ""
    users_dir = os.listdir(IMG_DIR)
    users_cnt = len(users_dir)

    # AKAZE検出器の生成
    akaze = cv2.AKAZE_create()
    # BFMatcherオブジェクトの生成q
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # CLAHEオブジェクトの生成
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))

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
        temp = 300

        user_id = int(user_dir)
        user_lst_id[index] = user_id

        files_dir = os.listdir(IMG_DIR+"/" + user_dir + "/")
        print("Processing ",user_dir)

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

            if target_des is None or comparing_des is None:
                break

            # BFMatcher で ライブラリに登録されている全ての画像に対してマッチングを行う
            matches = bf.match(target_des, comparing_des)
            if len(matches) == 0:
                break

            dist = [m.distance for m in matches]
            ret = sum(dist) / len(dist)
            if temp > ret:
                temp = ret
            print("{0} > {1}".format(file_dir,ret))

        user_lst_ret[index] = temp

        index = index + 1

    recognition_user = -1
    recognition_ret = 300

    for i in range(index):
        if user_lst_ret[i] < recognition_ret:
            recognition_ret = user_lst_ret[i]
            recognition_user = user_lst_id[i]
        print("User:", user_lst_id[i] ," Ret:", user_lst_ret[i])
    
    return recognition_user,recognition_ret

if __name__ == "__main__":
    main()
