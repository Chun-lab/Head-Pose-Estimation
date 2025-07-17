#coding:utf-8
from ultralytics import YOLO
import cv2
if __name__ == '__main__':
    # 所需加载的模型目录
    path = '/home/liuchun/data/ultralytics-main/yolov8n-face.pt'
    # 需要检测的图片地址
    img_path = "/home/liuchun/data/TokenHPE-main/TokenHPE-main/datasets/300W_LP/AFW/AFW_134212_1_0.jpg"

    model = YOLO(path, task='detect')
    # 检测图片
    results = model(img_path,conf=0.5)
    res = results[0].boxes.xyxy.tolist()

    img = cv2.imread(img_path)
    if res:
        x1, y1, x2, y2 = res[0][:4]
        face = img[int(y1):int(y2), int(x1):int(x2)]
    cv2.imwrite("/home/liuchun/data/TokenHPE-main/TokenHPE-main/face img/face.jpg", face)
