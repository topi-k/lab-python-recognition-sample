# -*- coding: utf-8 -*-
import argparse
import numpy as np

# OpenCVモジュールのインポート
import cv2

# DNNモデルのパス
PROTOTXT_PATH = './deploy.prototxt.txt'
WEIGHTS_PATH = './res10_300x300_ssd_iter_140000.caffemodel'

# 信頼度の閾値
CONFIDENCE = 0.5

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

def main():
    # 画像の読み込み
    img = imread(args.input)

    # モデルの読み込み
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, WEIGHTS_PATH)

    # 300x300に画像をリサイズ、画素値を調整
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))
    # 顔検出の実行
    net.setInput(blob)
    detections = net.forward()

    # 検出結果の可視化
    img_copy = img.copy()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONFIDENCE:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(img_copy, (startX, startY), (endX, endY),
                            (255, 0, 0), 2)

    # 結果の表示
    cv2.imshow('img', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # コマンドラインで指定された場合、実行結果を保存
    if args.output:
        cv2.imwrite(args.output, img_copy)
    return

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
        help="path to input image")
    ap.add_argument("-o", "--output", default=None,
            help="path to output image")
    args = ap.parse_args()

    main()