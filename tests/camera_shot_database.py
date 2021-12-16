# -*- coding: utf-8 -*-
# データセット用写真撮影
# 2021/12/16

import cv2
import dlib
import imutils
from imutils import face_utils
import numpy as np
import os
import time

# 顔検出用カメラ
DEVICE_ID = 0
capture_face = cv2.VideoCapture(DEVICE_ID)
# 耳検出用カメラ
DEVICE_ID = 1
capture_ear = cv2.VideoCapture(DEVICE_ID)

predictor_path = "./tests/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Haar-like特徴分類器の読み込み
ear_right_cascade = cv2.CascadeClassifier(
    'data/haarcascades/haarcascade_mcs_rightear.xml')
ear_left_cascade = cv2.CascadeClassifier(
    'data/haarcascades/haarcascade_mcs_leftear.xml')

# CLAHEオブジェクトの生成
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))

# 撮影レート
rate = 5
# 撮影される画像のマージン
margin = 10

while(1):
    user_id = input("UserID:")
    if user_id.isdecimal():
        if os.path.exists("./tests/dataset/"+user_id) == False:
            os.mkdir("./tests/dataset/"+user_id)
        if os.path.exists("./tests/dataset/"+user_id) == False:
            os.mkdir("./tests/dataset/"+user_id)
        user_id = int(user_id)
        break

while(True):
    ret_face, frame_face = capture_face.read()

    frame_face = imutils.resize(frame_face, width=1000)
    gray = cv2.cvtColor(frame_face, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    image_points = None

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (x, y) in shape:
            cv2.circle(frame_face, (x, y), 1, (255, 255, 255), -1)

        image_points = np.array([
                tuple(shape[30]),
                tuple(shape[21]),
                tuple(shape[22]),
                tuple(shape[39]),
                tuple(shape[42]),
                tuple(shape[31]),
                tuple(shape[35]),
                tuple(shape[48]),
                tuple(shape[54]),
                tuple(shape[57]),
                tuple(shape[8]),
                ],dtype='double')

    if len(rects) > 0:
        model_points = np.array([
                (0.0,0.0,0.0), # 30
                (-30.0,-125.0,-30.0), # 21
                (30.0,-125.0,-30.0), # 22
                (-60.0,-70.0,-60.0), # 39
                (60.0,-70.0,-60.0), # 42
                (-40.0,40.0,-50.0), # 31
                (40.0,40.0,-50.0), # 35
                (-70.0,130.0,-100.0), # 48
                (70.0,130.0,-100.0), # 54
                (0.0,158.0,-10.0), # 57
                (0.0,250.0,-50.0) # 8
                ])

        size = frame_face.shape

        focal_length = size[1]
        center = (size[1] // 2, size[0] // 2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype='double')

        dist_coeffs = np.zeros((4, 1))

        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
        mat = np.hstack((rotation_matrix, translation_vector))

        (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(mat)
        yaw = eulerAngles[1]
        pitch = eulerAngles[0]
        roll = eulerAngles[2]

        #print("yaw",int(yaw),"pitch",int(pitch),"roll",int(roll))

        cv2.putText(frame_face, 'yaw : ' + str(int(yaw)), (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(frame_face, 'pitch : ' + str(int(pitch)), (20, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(frame_face, 'roll : ' + str(int(roll)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        if (int(abs(yaw)) % rate == 0):
            if (int(abs(pitch)) > 5 or int(abs(pitch)) < -5):
                continue
            if (int(abs(roll)) < -5 or int(abs(roll)) > 5):
                continue

            ret_ear, frame_ear = capture_ear.read()
            # グレースケール変換 & コントラスト均等化
            frame_ear = cv2.cvtColor(frame_ear, cv2.COLOR_BGR2GRAY)
            frame_ear = clahe.apply(frame_ear)
            # 耳の検出
            ear_right = ear_right_cascade.detectMultiScale(frame_ear, scaleFactor=1.1, minNeighbors=1, minSize=(50,50))
            ear_left = ear_left_cascade.detectMultiScale(frame_ear, scaleFactor=1.1, minNeighbors=1, minSize=(50,50))
            
            if len(ear_right) != 0:
                for (ercx, ercy, ercw, erch) in ear_right:
                    save_img = frame_ear[ercy-margin:ercy+erch + margin, ercx-margin:ercx+ercw+margin]
                    print("saved ./tests/dataset/{0}/save_img_right_{1}_{0}_{2}.jpg".format(user_id, int(time.time()), int(yaw)))
                    cv2.imwrite("./tests/dataset/{0}/save_img_right_{1}_{0}_{2}.jpg".format(user_id, int(time.time()), int(yaw)), save_img) # 耳の画像を保存しておく
                    cv2.imwrite("./tests/dataset/{0}/save_img_right_{1}_{0}_{2}_raw_ear.jpg".format(user_id, int(time.time()), int(yaw)), frame_ear) # 切り取られる以前の画像
                    cv2.imwrite("./tests/dataset/{0}/save_img_right_{1}_{0}_{2}_raw_face.jpg".format(user_id, int(time.time()), int(yaw)), frame_face) # 切り取られる以前の画像
            if len(ear_left) != 0:
                for (elcx, elcy, elcw, elch) in ear_left:
                    save_img = frame_ear[elcy-margin:elcy+elch + margin, elcx-margin:elcx+elcw+margin]
                    print("saved ./tests/dataset/{0}/save_img_left_{1}_{0}_{2}.jpg".format(user_id, int(time.time()), int(yaw)))
                    cv2.imwrite("./tests/dataset/{0}/save_img_left_{1}_{0}_{2}.jpg".format(user_id, int(time.time()), int(yaw)), save_img) # 耳の画像を保存しておく
                    cv2.imwrite("./tests/dataset/{0}/save_img_left_{1}_{0}_{2}_raw_ear.jpg".format(user_id, int(time.time()), int(yaw)), frame_ear) # 切り取られる以前の画像
                    cv2.imwrite("./tests/dataset/{0}/save_img_left_{1}_{0}_{2}_raw_face.jpg".format(user_id, int(time.time()), int(yaw)), frame_face) # 切り取られる以前の画像

            cv2.imshow('frame_ear',frame_ear)

        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector,
                                                         translation_vector, camera_matrix, dist_coeffs)

        for p in image_points:
            cv2.drawMarker(frame_face, (int(p[0]), int(p[1])),  (0.0, 1.409845, 255),markerType=cv2.MARKER_CROSS, thickness=1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.arrowedLine(frame_face, p1, p2, (255, 0, 0), 2)

    cv2.imshow('frame_face',frame_face)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


capture_face.release()
capture_ear.release()

cv2.destroyAllWindows()
