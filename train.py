# import os
# import argparse
# import torch
# from torchvision import transforms
# import torch.backends.cudnn as cudnn
# import torch.nn as nn
# import datasets
# # from con import WeightedMatrix
# from loss import TokenGuideLoss
# from model import TokenHPE
# # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# def parse_args():
#     """Parse input arguments."""
#     parser = argparse.ArgumentParser(
#         description='Train TokenHPE model.')
#     parser.add_argument(
#         '--gpu', dest='gpu_id', help='GPU device id to use [0]',
#         default=0, type=int)
#     parser.add_argument(
#         '--num_epochs', dest='num_epochs',
#         help='Maximum number of training epochs.',
#         default=60, type=int)
#     parser.add_argument(
#         '--batch_size', dest='batch_size', help='Batch size.',
#         default=64, type=int)
#     parser.add_argument(
#         '--lr', dest='lr', help='Base learning rate.',
#         default=0.00001, type=float)
#     parser.add_argument(
#         '--dataset', dest='dataset', help='Dataset type. (BIWI/Pose_300W_LP)',
#         default='Pose_300W_LP', type=str)
#     parser.add_argument(
#         '--data_dir', dest='data_dir', help='Directory path for data.',
#         default='./datasets/300W_LP', type=str)
#     # examples
#     # 300W_LP dataset: './datasets/300W_LP'
#     # BIWI dataset: "./datasets/BIWI/BIWI_70_30_train.npz"
#     parser.add_argument(
#         '--filename_list', dest='filename_list',
#         help='Path to text file containing relative paths for every example.',
#         default='./datasets/300W_LP/files.txt', type=str)
#     # examples
#     # 300W_LP dataset: "./datasets/300W_LP/files.txt"
#     parser.add_argument(
#         '--alpha', dest='alpha', help='alpha in TokenGuideLoss.',
#         default=0.95, type=float)
#     # intermediate weights
#     parser.add_argument(
#         '--snapshot', dest='snapshot', help='Path of model snapshot.(xxx.tar format)',
#         default='', type=str)
#     # pretrained feature extractor weights (ViT)
#     parser.add_argument(
#         '--weights', dest='weights', help='Whether to use pretrained VIT-B/16 weights',
#         default='', type=str)
#     # examples  #/home/liuchun/data/TokenHPE-main/TokenHPE-main/jx_vit_base_patch16_224_in21k-e5005f0a.pth
#     # ./weights/vit_base_patch16_224_in21k.pth
#     parser.add_argument(
#         '--describe', dest='describe', help='Describe saving directory name.',
#         default='', type=str)
#     parser.add_argument(
#         '--output_string', dest='output_string',
#         help='String appended to output snapshots.', default='date_MM_DD', type=str)
#     args = parser.parse_args()
#     return args
#
# device = torch.device("cuda" if torch.cuda.is_available() else"cpu")
#
#
#
# learnable_matrix =torch.tensor ([[1.085, -1.10, -0.92],
#                     [-1.10, 1.12, 1.11],
#                     [-0.92, 1.11, 1.05]])
# learnable_matrix=  learnable_matrix.cuda()
# if __name__ == '__main__':
#
#     # w = nn.Parameter(torch.ones(9).to(device))
#
#     args = parse_args()
#     cudnn.enabled = True
#     num_epochs = args.num_epochs
#     batch_size = args.batch_size
#     gpu = args.gpu_id
#
#
#     if not os.path.exists('output/snapshots'):
#         os.makedirs('output/snapshots')
#
#     summary_name = '{}_{}_batch_size{}'.format(
#         'TokenHPE', args.describe, args.batch_size)
#
#     if not os.path.exists('output/snapshots/{}'.format(summary_name)):
#         os.makedirs('output/snapshots/{}'.format(summary_name))
#
#     model = TokenHPE(
#                  num_ori_tokens=11,
#                  depth=3,
#                  heads=8,
#                  embedding='sine',
#                  ViT_weights=args.weights,
#                  dim=128,
#                  )
#
#     if not args.snapshot == '':
#         saved_state_dict = torch.load(args.snapshot)
#         model.load_state_dict(saved_state_dict['model_state_dict'])
#         print("Intermediate weights used!")
#
#     model.to("cuda")
#     print('Loading data and preprocessing...')
#
#     normalize = transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225])
#
#     transformations = transforms.Compose([transforms.Resize(240),
#                                           transforms.RandomCrop(224),
#                                           transforms.ToTensor(),
#                                           normalize])
#
#     pose_dataset = datasets.getDataset(
#         args.dataset, args.data_dir, args.filename_list, transformations)
#
#     train_loader = torch.utils.data.DataLoader(
#         dataset=pose_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=1)
#
#     crit = TokenGuideLoss(alpha=args.alpha).cuda(gpu)
#
#     pg = [p for p in model.parameters() if p.requires_grad]
#     # opt = torch.optim.SGD(weighted_matrix.parameters(), lr=0.001)
#     optimizer = torch.optim.Adam(pg, lr=args.lr)
#
#     if not args.snapshot == '':
#         optimizer.load_state_dict(saved_state_dict['optimizer_state_dict'])
#
#     # learning rate decay
#     milestones = [20, 40]
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(
#         optimizer, milestones=milestones, gamma=0.5)
#
#
#     print('Starting training.')
#     for epoch in range(num_epochs):
#         loss_sum = .0
#         iter = 0
#         for i, (images, gt_mat, cont_labels, _) in enumerate(train_loader):
#             iter += 1
#             images = torch.Tensor(images).cuda(gpu)
#             # cont_labels: [batchsize, (pitch_agl, yaw_agl, roll_agl)]
#             # Forward pass
#             pred, ori_9_d = model(images)
#             # flipped_images = torch.flip(images, dims=[3])
#             # pred1, ori_9_d1 = model(flipped_images)
#             # pred1 = torch.matmul(pred1,learnable_matrix)
#             # pred =( pred + pred1 ) / 2
#
#
#             # pred:final prediction; dir_6_d: prediction on all orientations
#
#             overall_loss, pred_loss, ori_loss = crit(gt_mat.cuda(gpu), pred, cont_labels, ori_9_d)
#
#
#             optimizer.zero_grad()
#             overall_loss.backward()
#             optimizer.step()
#
#             # weighted_matrix.update(opt)
#
#             loss_sum += overall_loss.item()
#
#             if (i+1) % 5 == 0:
#                 print('Epoch [%d/%d],\t Iteration [%d/%d] \t Overall Loss: %.2f,\t Prediction Loss: %.5f,\t Orientation Loss: %.5f.'
#                      % (
#                           epoch+1,
#                           num_epochs,
#                           i+1,
#                           len(pose_dataset)//batch_size,
#                           overall_loss.item(),
#                           pred_loss.item(),
#                           ori_loss.item(),
#                       )
#                       )
#
#         scheduler.step()
#
#         # Save models at numbered epochs.
#         if epoch % 1 == 0 and epoch < num_epochs:
#             print('Taking snapshot...',
#                   torch.save({
#                       'epoch': epoch,
#                       'model_state_dict': model.state_dict(),
#                       'optimizer_state_dict': optimizer.state_dict(),
#                   }, 'output/snapshots/' + summary_name + '/' + args.output_string +
#                       '_epoch_' + str(epoch+1) + '.tar')
#                   )
# import os
# import argparse
# import torch
# from torchvision import transforms
# import torch.backends.cudnn as cudnn
# import datasets
# from loss import TokenGuideLoss
# from model import TokenHPE
#
#
# def parse_args():
#     """Parse input arguments."""
#     parser = argparse.ArgumentParser(
#         description='Train TokenHPE model.')
#     parser.add_argument(
#         '--gpu', dest='gpu_id', help='GPU device id to use [0]',
#         default=0, type=int)
#     parser.add_argument(
#         '--num_epochs', dest='num_epochs',
#         help='Maximum number of training epochs.',
#         default=60, type=int)
#     parser.add_argument(
#         '--batch_size', dest='batch_size', help='Batch size.',
#         default=64, type=int)
#     parser.add_argument(
#         '--lr', dest='lr', help='Base learning rate.',
#         default=0.00001, type=float)
#     parser.add_argument(
#         '--dataset', dest='dataset', help='Dataset type. (BIWI/Pose_300W_LP)',
#         default='Pose_300W_LP', type=str)
#     parser.add_argument(
#         '--data_dir', dest='data_dir', help='Directory path for data.',
#         default='./datasets/300W_LP', type=str)
#     # examples
#     # 300W_LP dataset: './datasets/300W_LP'
#     # BIWI dataset: "./datasets/BIWI/BIWI_70_30_train.npz"
#     parser.add_argument(
#         '--filename_list', dest='filename_list',
#         help='Path to text file containing relative paths for every example.',
#         default='./datasets/300W_LP/files.txt', type=str)
#     # examples
#     # 300W_LP dataset: "./datasets/300W_LP/files.txt"
#     parser.add_argument(
#         '--alpha', dest='alpha', help='alpha in TokenGuideLoss.',
#         default=0.95, type=float)
#     # intermediate weights
#     parser.add_argument(
#         '--snapshot', dest='snapshot', help='Path of model snapshot.(xxx.tar format)',
#         default='', type=str)
#     # pretrained feature extractor weights (ViT)
#     parser.add_argument(
#         '--weights', dest='weights', help='Whether to use pretrained VIT-B/16 weights',
#         default='', type=str)
#     # examples
#     # ./weights/vit_base_patch16_224_in21k.pth
#     parser.add_argument(
#         '--describe', dest='describe', help='Describe saving directory name.',
#         default='', type=str)
#     parser.add_argument(
#         '--output_string', dest='output_string',
#         help='String appended to output snapshots.', default='date_MM_DD', type=str)
#     args = parser.parse_args()
#     return args
#
#
# if __name__ == '__main__':
#
#     args = parse_args()
#     cudnn.enabled = True
#     num_epochs = args.num_epochs
#     batch_size = args.batch_size
#     gpu = args.gpu_id
#
#
#     if not os.path.exists('output/snapshots'):
#         os.makedirs('output/snapshots')
#
#     summary_name = '{}_{}_batch_size{}'.format(
#         'TokenHPE', args.describe, args.batch_size)
#
#     if not os.path.exists('output/snapshots/{}'.format(summary_name)):
#         os.makedirs('output/snapshots/{}'.format(summary_name))
#     # 新路径示例
#     # base_path = '/data/liuchun'
#
#     # if not os.path.exists(base_path):
#     #     os.makedirs(base_path)
#     #
#     # summary_name = '{}_{}_batch_size{}'.format(
#     #     'TokenHPE', args.describe, args.batch_size)
#     #
#     # if not os.path.exists(os.path.join(base_path, summary_name)):
#     #     os.makedirs(os.path.join(base_path, summary_name))
#
#     model = TokenHPE(
#                  num_ori_tokens=11,
#                  depth=3,
#                  heads=8,
#                  embedding='sine',
#                  ViT_weights=args.weights,
#                  dim=128,
#                  )
#
#     if not args.snapshot == '':
#         saved_state_dict = torch.load(args.snapshot)
#         model.load_state_dict(saved_state_dict['model_state_dict'])
#         print("Intermediate weights used!")
#
#     model.to("cuda")
#     print('Loading data and preprocessing...')
#
#     normalize = transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225])
#
#     transformations = transforms.Compose([transforms.Resize(240),
#                                           transforms.RandomCrop(224),
#                                           transforms.ToTensor(),
#                                           normalize])
#
#     pose_dataset = datasets.getDataset(
#         args.dataset, args.data_dir, args.filename_list, transformations)
#
#     train_loader = torch.utils.data.DataLoader(
#         dataset=pose_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=1)
#
#     crit = TokenGuideLoss(alpha=args.alpha).cuda(gpu)
#
#     pg = [p for p in model.parameters() if p.requires_grad]
#     optimizer = torch.optim.Adam(pg, lr=args.lr)
#
#     if not args.snapshot == '':
#         optimizer.load_state_dict(saved_state_dict['optimizer_state_dict'])
#
#     # learning rate decay
#     milestones = [20, 40]
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(
#         optimizer, milestones=milestones, gamma=0.5)
#
#
#     print('Starting training.')
#     for epoch in range(num_epochs):
#         loss_sum = .0
#         iter = 0
#         for i, (images, gt_mat, cont_labels, _) in enumerate(train_loader):
#             iter += 1
#             images = torch.Tensor(images).cuda(gpu)
#             # cont_labels: [batchsize, (pitch_agl, yaw_agl, roll_agl)]
#             # Forward pass
#             pred, ori_9_d = model(images)
#             # pred:final prediction; dir_6_d: prediction on all orientations
#
#             overall_loss, pred_loss, ori_loss = crit(gt_mat.cuda(gpu), pred, cont_labels, ori_9_d)
#
#             optimizer.zero_grad()
#             overall_loss.backward()
#             optimizer.step()
#
#             loss_sum += overall_loss.item()
#
#             if (i+1) % 20 == 0:
#                 print('Epoch [%d/%d],\t Iteration [%d/%d] \t Overall Loss: %.2f,\t Prediction Loss: %.5f,\t Orientation Loss: %.5f.'
#                      % (
#                           epoch+1,
#                           num_epochs,
#                           i+1,
#                           len(pose_dataset)//batch_size,
#                           overall_loss.item(),
#                           pred_loss.item(),
#                           ori_loss.item(),
#                       )
#                       )
#
#         scheduler.step()
#
#         if epoch % 1 == 0 and epoch < num_epochs:
#             print('Taking snapshot...',
#                   torch.save({
#                       'epoch': epoch,
#                       'model_state_dict': model.state_dict(),
#                       'optimizer_state_dict': optimizer.state_dict(),
#                   }, 'output/snapshots/' + summary_name + '/' + args.output_string +
#                       '_epoch_' + str(epoch+1) + '.tar')
#                   )
#
#         # Save models at numbered epochs.

import os
import argparse
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
import datasets
from loss import TokenGuideLoss
from model import TokenHPE
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Train TokenHPE model.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--num_epochs', dest='num_epochs',
        help='Maximum number of training epochs.',
        default=60, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=64, type=int)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.00001, type=float)
    parser.add_argument(
        '--dataset', dest='dataset', help='Dataset type. (BIWI/Pose_300W_LP)',
        default='Pose_300W_LP', type=str)
    parser.add_argument(
        '--data_dir', dest='data_dir', help='Directory path for data.',
        default='./datasets/300W_LP', type=str)
    # examples
    # 300W_LP dataset: './datasets/300W_LP'
    # BIWI dataset: "./datasets/BIWI/BIWI_70_30_train.npz"
    parser.add_argument(
        '--filename_list', dest='filename_list',
        help='Path to text file containing relative paths for every example.',
        default='./datasets/300W_LP/files.txt', type=str)
    # examples
    # 300W_LP dataset: "./datasets/300W_LP/files.txt"
    parser.add_argument(
        '--alpha', dest='alpha', help='alpha in TokenGuideLoss.',
        default=0.95, type=float)
    # intermediate weights
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.(xxx.tar format)',
        default='', type=str)
    # pretrained feature extractor weights (ViT)
    parser.add_argument(
        '--weights', dest='weights', help='Whether to use pretrained VIT-B/16 weights',
        default='', type=str)
    # examples
    # ./weights/vit_base_patch16_224_in21k.pth
    parser.add_argument(
        '--describe', dest='describe', help='Describe saving directory name.',
        default='', type=str)
    parser.add_argument(
        '--output_string', dest='output_string',
        help='String appended to output snapshots.', default='date_MM_DD', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    summary_name = '{}_{}_batch_size{}'.format(
        'TokenHPE', args.describe, args.batch_size)

    if not os.path.exists('output/snapshots/{}'.format(summary_name)):
        os.makedirs('output/snapshots/{}'.format(summary_name))
    model = TokenHPE(
        num_ori_tokens=11,
        depth=3,
        heads=8,
        embedding='sine',
        ViT_weights=args.weights,
        dim=128,
    )

    if not args.snapshot == '':
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict['model_state_dict'])
        print("Intermediate weights used!")

    model.to("cuda")
    print('Loading data and preprocessing...')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    transformations = transforms.Compose([transforms.Resize(240),
                                          transforms.RandomCrop(224),
                                          transforms.ToTensor(),
                                          normalize])

    pose_dataset = datasets.getDataset(
        args.dataset, args.data_dir, args.filename_list, transformations)

    train_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1)

    crit = TokenGuideLoss(alpha=args.alpha).cuda(gpu)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(pg, lr=args.lr)

    if not args.snapshot == '':
        optimizer.load_state_dict(saved_state_dict['optimizer_state_dict'])

    # learning rate decay
    milestones = [20, 40]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5)

    print('Starting training.')
    for epoch in range(num_epochs):
        loss_sum = .0
        iter = 0
        for i, (images, gt_mat, cont_labels, _) in enumerate(train_loader):
            iter += 1
            images = torch.Tensor(images).cuda(gpu)
            # cont_labels: [batchsize, (pitch_agl, yaw_agl, roll_agl)]
            # Forward pass
            pred, ori_9_d = model(images)
            # pred:final prediction; dir_6_d: prediction on all orientations

            overall_loss, pred_loss, ori_loss, constanct_loss ,our_loss= crit(gt_mat.cuda(gpu), pred, cont_labels, ori_9_d)
            # overall_loss, pred_loss, ori_loss, constanct_loss = crit(gt_mat.cuda(gpu), pred, cont_labels,
            #                                                                    ori_9_d)
            # overall_loss, pred_loss, ori_loss= crit(gt_mat.cuda(gpu), pred, cont_labels, ori_9_d)
            optimizer.zero_grad()
            overall_loss.backward()
            optimizer.step()

            loss_sum += overall_loss.item()

            if (i + 1) % 20 == 0:
                print(
                    'Epoch [%d/%d],\t Iteration [%d/%d] \t Overall Loss: %.2f,\t Prediction Loss: %.5f,\t Orientation Loss: %.5f,\t constanct_loss: %.5f,\t our_Loss: %.5f.'
                    # 'Epoch [%d/%d],\t Iteration [%d/%d] \t Overall Loss: %.2f,\t Prediction Loss: %.5f,\t Orientation Loss: %.5f.'
                    # 'Epoch [%d/%d],\t Iteration [%d/%d] \t Overall Loss: %.2f,\t Prediction Loss: %.5f,\t Orientation Loss: %.5f,\t constanct_loss: %.5f.'
                    % (
                        epoch + 1,
                        num_epochs,
                        i + 1,
                        len(pose_dataset) // batch_size,
                        overall_loss.item(),
                        pred_loss.item(),
                        ori_loss.item(),
                        constanct_loss.item(),
                        our_loss.item(),
                    )
                    )

        scheduler.step()

        if epoch % 1 == 0 and epoch < num_epochs:
            print('Taking snapshot...',
                  torch.save({
                      'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                  }, 'output/snapshots/' + summary_name + '/' + args.output_string +
                     '_epoch_' + str(epoch + 1) + '.tar')
                  )

        # Save models at numbered epochs.
