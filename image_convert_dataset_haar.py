# -*- coding: utf-8 -*-
# AMI Ear Database をデータセットとして登録するプログラム
# 2021/12/01

import cv2
import sys
import time
import os
import re
import shutil

# Haar-like特徴分類器の読み込み
face_cascade = cv2.CascadeClassifier(
    'data/haarcascades/haarcascade_frontalface_default.xml')

ear_right_cascade = cv2.CascadeClassifier(
    'data/haarcascades/haarcascade_mcs_rightear.xml')
ear_left_cascade = cv2.CascadeClassifier(
    'data/haarcascades/haarcascade_mcs_leftear.xml')
# Settings
ami_database_dir = os.path.abspath(os.path.dirname(__file__)) + '/tests/ami_ear_database/subset-1/'""
ami_database_dir_list = os.listdir(ami_database_dir)
pattern = "([0-9]+)_(.*)_ear.jpg"
margin = 10

save_img_cnt = 0

for dir in ami_database_dir_list:
    result = re.match(pattern, dir)
    user_id = int(result.group(1))
    if result:
        img = cv2.imread(ami_database_dir + dir)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ear_right = ear_right_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(100,100))
        ear_left = ear_left_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(100,100))
        if len(ear_right) != 0:
            if os.path.exists("./images/right/" + str(user_id)) == False:
                os.mkdir("./images/right/" + str(user_id))
            for (ercx, ercy, ercw, erch) in ear_right:
                img = img[ercy-margin:ercy+erch + margin, ercx-margin:ercx+ercw+margin]
                cv2.imwrite("./images/right/{0}/{1}".format(user_id, dir), img)
                print("Save["+str(user_id)+"] >> ./images/right/{0}/{1}".format(user_id, dir))
        if len(ear_left) != 0:
            if os.path.exists("./images/left/" + str(user_id)) == False:
                os.mkdir("./images/left/" + str(user_id))
            for (ercx, ercy, ercw, erch) in ear_left:
                img = img[ercy-margin:ercy+erch + margin, ercx-margin:ercx+ercw+margin]
                cv2.imwrite("./images/left/{0}/{1}".format(user_id, dir), img)
                print("Save["+str(user_id)+"] >> ./images/left/{0}/{1}".format(user_id, dir))