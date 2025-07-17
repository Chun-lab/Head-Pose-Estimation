import argparse
import cv2
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
import utils
import matplotlib
import numpy as np
from ultralytics import YOLO
import seaborn as sns
from PIL import Image
from model import TokenHPE
sns.set()

matplotlib.use('TkAgg')

def scale(yaw,pitch,roll):
    if yaw>20 or yaw<-20:
        return 3
    elif pitch >-30:
        return 1
    else:
        return 2

if __name__ == '__main__':
    # 所需加载的模型目录
    path = '/data/liuchun/data/ultralytics-main/face/runs/detect/train/weights/runs/detect/train4/weights/best.pt'
    # / data / liuchun / data / ultralytics - main / face / runs / detect / train / weights / best.pt
    # path = '/home/liuchun/data/TokenHPE-main/TokenHPE-main/yolov8n-face.pt'
    # 需要检测的图片地址

    img_path = "/data/liuchun/data/TokenHPE-main/TokenHPE-main/img_000031.jpg"
    model_path = '/data/liuchun/data/TokenHPE-main/TokenHPE-main/baocun /end_170_11.tar'

    detect = YOLO(path, task='detect')
    # 检测图片
    results = detect(img_path,conf=0.5)
    res = results[0].boxes.xyxy.tolist()
    transformations = transforms.Compose([transforms.Resize(250),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    estimate = TokenHPE(num_ori_tokens=11,
                     depth=3, heads=8, embedding='sine', dim=128, inference_view=False
                     ).to("cuda")
    img = cv2.imread(img_path)
    if model_path != "":
        saved_state_dict = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in saved_state_dict:
            estimate.load_state_dict(saved_state_dict['model_state_dict'])
            print("model weight loaded!")
        else:
            estimate.load_state_dict(saved_state_dict)
    else:
        print("model weight failed!")

    estimate.to("cuda")
    shang=0
    xia=0
    zong=0
    # Test the Model
    estimate.eval()
    scale_factor = 1.2
    with torch.no_grad():
        for i, each in enumerate(res):
            zong=zong+1
            x1, y1, x2, y2 = each[:4]
            c1, d1, c2, d2 = each[:4]
            width = x2 - x1
            height = y2 - y1

            # 计算扩大后的宽度和高度
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # 计算扩大后的人脸框的左上角和右下角坐标
            x1 -= int((new_width - width) / 2)  -5
            y1 -= int((new_height - height) / 2) -5
            x2 = x1 + new_width
            y2 = y1 + new_height

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            c1 = int(c1)
            d1 = int(d1)
            c2 = int(c2)
            d2 = int(d2)
            output_img = img[y1:y2, x1:x2]
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            face_img = img_pil.crop((x1,y1,x2,y2))
            face_img = transformations(face_img)
            face_img = torch.unsqueeze(face_img, dim=0)
            face_img = torch.Tensor(face_img).to("cuda")
            R_pred, ori_9_d = estimate(face_img)
            euler = utils.compute_euler_angles_from_rotation_matrices(
                R_pred) * 180 / np.pi
            p_pred_deg = euler[:, 0].cpu()
            y_pred_deg = euler[:, 1].cpu()
            r_pred_deg = euler[:, 2].cpu()
            c=scale(y_pred_deg,p_pred_deg,r_pred_deg)
            print(i)
            print(y_pred_deg,p_pred_deg,r_pred_deg)
            if c==1:
                shang+=1
            else:
                xia+=1
            # print(f"Prediction: pitch:{p_pred_deg[0]:.2f}, yaw:{y_pred_deg[0]:.2f}, roll:{r_pred_deg[0]:.2f}.")

            utils.draw_axis(output_img , y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], size=30)
            img[y1:y2, x1:x2] = output_img# tdx=150, tdy=150,
            # cv2.rectangle(img, (c1, d1), (c2, d2), (0, 255, 0), 2)
            text = f"{c}"
            print(x1,y1,x2,y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # cv2.putText(img, text, (x1, y1 ), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        print("zong",zong)

            # 显示裁剪后的区域
        cv2.imshow('result', img)
        cv2.waitKey(0)
