# -*- coding: utf-8 -*-
# AMI Ear Databaseの耳介を登録画像として登録するプログラム
# 2021/12/01

import cv2
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
    if result:
        if result.group(2) == "back":
            user_id = str(int(result.group(1)))
            if os.path.exists("./ami_images/right/" + user_id) == False:
                os.mkdir("./ami_images/right/" + user_id)
            shutil.copy(ami_database_dir + dir , "./ami_images/right/" + user_id + "/" + dir)
            print ("Copy "+ ami_database_dir + "/" + dir + " to " "./ami_images/right/" + user_id + "/" + dir)
        elif result.group(2) == "front":
            user_id = str(int(result.group(1)))
            if os.path.exists("./ami_images/left/" + user_id) == False:
                os.mkdir("./ami_images/left/" + user_id)
            shutil.copy(ami_database_dir + dir , "./ami_images/left/" + user_id + "/" + dir)
            print ("Copy "+ ami_database_dir + "/" + dir + " to " "./ami_images/left/" + user_id + "/" + dir)          
