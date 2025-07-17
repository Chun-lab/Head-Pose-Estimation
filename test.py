# import os
# import argparse
# import numpy as np
# import cv2
# import torch
# from torchvision import transforms
# import torch.backends.cudnn as cudnn
# import datasets
# import utils
# import matplotlib
# from model import TokenHPE
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# matplotlib.use('TkAgg')
#
#
# def parse_args():
#     """Parse input arguments."""
#     parser = argparse.ArgumentParser(
#         description='Test TokenHPE model.')
#     parser.add_argument('--gpu',
#                         dest='gpu_id', help='GPU device id to use [0]',
#                         default=0, type=int)
#     parser.add_argument('--data_dir',
#                         dest='data_dir', help='Directory path for data.',
#                         default='', type=str)
#     # examples
#     # ./datasets/AFLW2000
#     # ./datasets/BIWI/BIWI.npz
#     parser.add_argument('--filename_list',
#                         dest='filename_list',
#                         help='Path to text file containing relative paths for every example.',
#                         default='', type=str)
#     # examples
#     # ./datasets/AFLW2000/files.txt
#     # ./datasets/BIWI/BIWI.npz
#     parser.add_argument('--model_path',
#                         dest='model_path', help='model path.',
#                         default='./weights/TokenHPEv1-ViTB-224_224-lyr3.tar', type=str)
#     parser.add_argument('--batch_size',
#                         dest='batch_size', help='Batch size.',
#                         default=32, type=int)
#     parser.add_argument('--show_viz',
#                         dest='show_viz', help='Save images with pose cube.',
#                         default=True, type=bool)
#     parser.add_argument('--dataset',
#                         dest='dataset', help='Dataset type.(AFLW2000/BIWI)',
#                         default='AFLW2000', type=str)
#
#     args = parser.parse_args()
#     return args
#
# def load_filtered_state_dict(model, snapshot):
#     # By user apaszke from discuss.pytorch.org
#     model_dict = model.state_dict()
#     snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
#     model_dict.update(snapshot)
#     model.load_state_dict(model_dict)
#
#
# if __name__ == '__main__':
#     args = parse_args()
#     cudnn.enabled = True
#     gpu = args.gpu_id
#     model_path = args.model_path
#
#
#     model = TokenHPE(num_ori_tokens=11,
#                  depth=3, heads=8, embedding='sine', dim=128
#                  ).to("cuda")
#
#
#     print('Loading data.')
#
#     transformations = transforms.Compose([transforms.Resize(250),
#                                           transforms.CenterCrop(224),
#                                           transforms.ToTensor(),
#                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#
#     pose_dataset = datasets.getDataset(
#         args.dataset, args.data_dir, args.filename_list, transformations, train_mode=False)
#     test_loader = torch.utils.data.DataLoader(
#         dataset=pose_dataset,
#         batch_size=args.batch_size,
#         num_workers=2)
#
#     # Load snapshot
#     saved_state_dict = torch.load(model_path, map_location='cpu')
#     if 'model_state_dict' in saved_state_dict:
#         model.load_state_dict(saved_state_dict['model_state_dict'])
#         print("model weight loaded!")
#     else:
#         model.load_state_dict(saved_state_dict)
#
#     model.cuda(gpu)
#
#     # Test the Model
#     model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
#
#     total = 0
#     yaw_error = pitch_error = roll_error = .0
#     v1_err = v2_err = v3_err = .0
#
#     with torch.no_grad():
#
#         for i, (images, r_label, cont_labels, name) in enumerate(test_loader):
#             images = torch.Tensor(images).cuda(gpu)
#             total += cont_labels.size(0)
#
#             # gt matrix
#             R_gt = r_label
#             # print("R_gt",R_gt)
#             # gt euler
#             y_gt_deg = cont_labels[:, 0].float() * 180 / np.pi # yaw
#             p_gt_deg = cont_labels[:, 1].float() * 180 / np.pi # pitch
#             r_gt_deg = cont_labels[:, 2].float() * 180 / np.pi # row
#             flipped_images = torch.flip(images, dims=[3])
#             R_pred, ori_9_d = model(images)
#             # print("R_pred",R_pred)
#             # print("Ori_9d", ori_9_d)
#             R_pred1, ori_9_d1 = model(flipped_images)
#
#             weight1 = torch.tensor([[ 1.0855, -1.1001, -0.9196],
#                                     [-1.0997,  1,  1.1461],
#                                     [-0.9204,  1.1451,  1.0862]]).cuda()
#             #
#             # weight1 = torch.tensor([[1.085, -1.10, -0.92],
#             #            [-1.10, 1.12, 1.11],
#             #            [-0.92, 1.11, 1.05]]).cuda()
#
#             R_pred1 = R_pred1 * weight1
#             R_pred = (R_pred1+R_pred)/2
#
#             euler1 = utils.compute_euler_angles_from_rotation_matrices(
#                 R_pred) * 180 / np.pi
#             euler2 = utils.compute_euler_angles_from_rotation_matrices(
#                 R_pred1) * 180 / np.pi
#             # p_pred_deg = (euler1[:, 0].cpu()+euler2[:, 0].cpu())/2
#             # y_pred_deg = (euler1[:, 1].cpu()-euler2[:, 1].cpu())/2
#             # r_pred_deg = (euler1[:, 2].cpu()-euler2[:, 2].cpu())/2
#
#             euler = utils.compute_euler_angles_from_rotation_matrices(
#                 R_pred) * 180 / np.pi
#             p_pred_deg = euler[:, 0].cpu()
#             y_pred_deg = euler[:, 1].cpu()
#             r_pred_deg = euler[:, 2].cpu()
#
#             R_pred = R_pred.cpu()
#             v1_err += torch.sum(torch.acos(torch.clamp(
#                 torch.sum(R_gt[:, 0] * R_pred[:, 0], 1), -1, 1)) * 180 / np.pi)
#             v2_err += torch.sum(torch.acos(torch.clamp(
#                 torch.sum(R_gt[:, 1] * R_pred[:, 1], 1), -1, 1)) * 180 / np.pi)
#             v3_err += torch.sum(torch.acos(torch.clamp(
#                 torch.sum(R_gt[:, 2] * R_pred[:, 2], 1), -1, 1)) * 180 / np.pi)
#
#             pitch_error += torch.sum(torch.min(
#                 torch.stack((torch.abs(p_gt_deg - p_pred_deg), torch.abs(p_pred_deg + 360 - p_gt_deg), torch.abs(
#                     p_pred_deg - 360 - p_gt_deg), torch.abs(p_pred_deg + 180 - p_gt_deg),
#                              torch.abs(p_pred_deg - 180 - p_gt_deg))), 0)[0])
#             yaw_error += torch.sum(torch.min(
#                 torch.stack((torch.abs(y_gt_deg - y_pred_deg), torch.abs(y_pred_deg + 360 - y_gt_deg), torch.abs(
#                     y_pred_deg - 360 - y_gt_deg), torch.abs(y_pred_deg + 180 - y_gt_deg),
#                              torch.abs(y_pred_deg - 180 - y_gt_deg))), 0)[0])
#             roll_error += torch.sum(torch.min(
#                 torch.stack((torch.abs(r_gt_deg - r_pred_deg), torch.abs(r_pred_deg + 360 - r_gt_deg), torch.abs(
#                     r_pred_deg - 360 - r_gt_deg), torch.abs(r_pred_deg + 180 - r_gt_deg),
#                              torch.abs(r_pred_deg - 180 - r_gt_deg))), 0)[0])
#
#
#         print('Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f' % (
#             yaw_error / total, pitch_error / total, roll_error / total,
#             (yaw_error + pitch_error + roll_error) / (total * 3)))
#
#         print('Vec1: %.4f, Vec2: %.4f, Vec3: %.4f, VMAE: %.4f' % (
#             v1_err / total, v2_err / total, v3_err / total,
#             (v1_err + v2_err + v3_err) / (total * 3)))
import os
import argparse
# from models.t2t_vit import T2T_module
import numpy as np
import cv2
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
import datasets
import utils
import matplotlib
from PIL import Image
from model import TokenHPE
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
matplotlib.use('TkAgg')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Test TokenHPE model.')
    parser.add_argument('--gpu',
                        dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--data_dir',
                        dest='data_dir', help='Directory path for data.',
                        default='', type=str)
    # examples
    # ./datasets/AFLW2000
    # ./datasets/BIWI/BIWI.npz
    parser.add_argument('--filename_list',
                        dest='filename_list',
                        help='Path to text file containing relative paths for every example.',
                        default='', type=str)
    # examples
    # ./datasets/AFLW2000/files.txt
    # ./datasets/BIWI/BIWI.npz
    parser.add_argument('--model_path',
                        dest='model_path', help='model path.',
                        default='/home/liuchun/data/TokenHPE-main/TokenHPE-main/weights/TokenHPEv1-ViTB-224_224-lyr3.tar', type=str)
    parser.add_argument('--batch_size',
                        dest='batch_size', help='Batch size.',
                        default=32, type=int)
    parser.add_argument('--show_viz',
                        dest='show_viz', help='Save images with pose cube.',
                        default=True, type=bool)
    parser.add_argument('--dataset',
                        dest='dataset', help='Dataset type.(AFLW2000/BIWI)',
                        default='AFLW2000', type=str)

    args = parser.parse_args()
    return args


def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = args.gpu_id
    model_path = args.model_path


    model = TokenHPE(num_ori_tokens=11,
                 depth=3, heads=8, embedding='sine', dim=128
                 ).to("cuda")


    print('Loading data.')

    transformations = transforms.Compose([transforms.Resize(240),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # resize=340 BIWI


    pose_dataset = datasets.getDataset(
        args.dataset, args.data_dir, args.filename_list, transformations, train_mode=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=args.batch_size,
        num_workers=2)

    # Load snapshot
    saved_state_dict = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
        print("model weight loaded!")
    else:
        model.load_state_dict(saved_state_dict)

    model.cuda(gpu)

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    total = 0
    yaw_error = pitch_error = roll_error = .0
    v1_err = v2_err = v3_err = .0

    with torch.no_grad():

        for i, (images, r_label, cont_labels, name) in enumerate(test_loader):
            images = torch.Tensor(images).cuda(gpu)
            total += cont_labels.size(0)

            # gt matrix
            R_gt = r_label

            # gt euler
            y_gt_deg = cont_labels[:, 0].float() * 180 / np.pi # yaw
            p_gt_deg = cont_labels[:, 1].float() * 180 / np.pi # pitch
            # p_gt_deg = cont_labels[:, 0].float() * 180 / np.pi  # yaw
            # y_gt_deg = cont_labels[:, 1].float() * 180 / np.pi  # pitch
            r_gt_deg = cont_labels[:, 2].float() * 180 / np.pi # row
            # print("000000000000000000000000000",y_gt_deg,p_gt_deg,r_gt_deg)
            # flipped_images = torch.flip(images, dims=[3])
            # R_pred1, ori_9_d1 = model(flipped_images)
            R_pred, ori_9_d = model(images)
            print(R_pred)
            # print('p_gt_deg:',p_gt_deg,'y_gt_deg:',y_gt_deg,'r_gt_deg:',r_gt_deg)
            # euler1 = utils.compute_euler_angles_from_rotation_matrices(
            #     R_pred) * 180 / np.pi
            # euler2 = utils.compute_euler_angles_from_rotation_matrices(
            #     R_pred1) * 180 / np.pi
            # p_pred_deg = (euler1[:, 0].cpu()+euler2[:, 0].cpu())/2
            # y_pred_deg = (euler1[:, 1].cpu()-euler2[:, 1].cpu())/2
            # r_pred_deg = (euler1[:, 2].cpu()-euler2[:, 2].cpu())/2
            # print('p_pred_deg:', p_pred_deg, 'y_pred_deg:', y_pred_deg, 'r_pred_deg:', r_pred_deg)
            # print(R_pred)
            euler = utils.compute_euler_angles_from_rotation_matrices(
                    R_pred) * 180 / np.pi
            p_pred_deg = euler[:, 0].cpu()
            y_pred_deg = euler[:, 1].cpu()
            r_pred_deg = euler[:, 2].cpu()

            R_pred = R_pred.cpu()

            v1_err += torch.sum(torch.acos(torch.clamp(
                torch.sum(R_gt[:, 0] * R_pred[:, 0], 1), -1, 1)) * 180 / np.pi)
            v2_err += torch.sum(torch.acos(torch.clamp(
                torch.sum(R_gt[:, 1] * R_pred[:, 1], 1), -1, 1)) * 180 / np.pi)
            v3_err += torch.sum(torch.acos(torch.clamp(
                torch.sum(R_gt[:, 2] * R_pred[:, 2], 1), -1, 1)) * 180 / np.pi)

            pitch_error += torch.sum(torch.min(
                torch.stack((torch.abs(p_gt_deg - p_pred_deg), torch.abs(p_pred_deg + 360 - p_gt_deg), torch.abs(
                    p_pred_deg - 360 - p_gt_deg), torch.abs(p_pred_deg + 180 - p_gt_deg),
                             torch.abs(p_pred_deg - 180 - p_gt_deg))), 0)[0])
            yaw_error += torch.sum(torch.min(
                torch.stack((torch.abs(y_gt_deg - y_pred_deg), torch.abs(y_pred_deg + 360 - y_gt_deg), torch.abs(
                    y_pred_deg - 360 - y_gt_deg), torch.abs(y_pred_deg + 180 - y_gt_deg),
                             torch.abs(y_pred_deg - 180 - y_gt_deg))), 0)[0])
            roll_error += torch.sum(torch.min(
                torch.stack((torch.abs(r_gt_deg - r_pred_deg), torch.abs(r_pred_deg + 360 - r_gt_deg), torch.abs(
                    r_pred_deg - 360 - r_gt_deg), torch.abs(r_pred_deg + 180 - r_gt_deg),
                             torch.abs(r_pred_deg - 180 - r_gt_deg))), 0)[0])

        print('Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f' % (
            yaw_error / total, pitch_error / total, roll_error / total,
            (yaw_error + pitch_error + roll_error) / (total * 3)))

        print('Vec1: %.4f, Vec2: %.4f, Vec3: %.4f, VMAE: %.4f' % (
            v1_err / total, v2_err / total, v3_err / total,
            (v1_err + v2_err + v3_err) / (total * 3)))



