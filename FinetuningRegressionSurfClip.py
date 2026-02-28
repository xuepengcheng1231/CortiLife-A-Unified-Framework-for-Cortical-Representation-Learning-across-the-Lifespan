# Based on SLIP code bases
# https://github.com/facebookresearch/SLIP
# --------------------------------------------------------'
import argparse
import re
from collections import OrderedDict
import json

import clip
import math
import os
import sys
import time


from timm.models import load_pretrained

import CARZERO
from Dataset import DataAugmentationForMAE_parcel, SurfaceFolder, Finetune_SurfaceFolder

# import subprocess

try:
    import wandb
except ImportError:
    wandb = None

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import SurfClip
from datasets import get_dataset
from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix,
                             roc_curve, precision_recall_curve, auc,
                             precision_score, recall_score)
import models
from tokenizer import SimpleTokenizer
from utils import AverageMeter, ProgressMeter, accuracy
import utils
from torchvision.datasets import ImageFolder
from utils import GaussianBlur, Solarize
from losses import DetailCLIPLoss, get_metric_names, CLIPloss, SurfCliploss, get_surfclip_names, \
    get_finetuning_surfclip_names
import torch.distributed as dist
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def get_args_parser():
    parser = argparse.ArgumentParser(description='DetailCLIP pre-training and evaluation', add_help=False)
    # Data
    parser.add_argument('--root_path', default='/data2/member/SurfClip/data', type=str)
    parser.add_argument('--save_path', default='/data2/member/SurfClip/data/HCP', type=str)
    parser.add_argument('--surf_name', default='HCP.npy', type=str)
    parser.add_argument('--label_name', default='HCP_label.npy', type=str)
    parser.add_argument('--finetune', default='/data2/member/SurfClip/Output_CLIP/No_HCP_CLIP/checkpoint_10.pt', type=str)
    # parser.add_argument('--global_mean', default="/data2/member/SurfClip/data_surfclip/No_HCP_mean.npy", type=str)
    # parser.add_argument('--global_std', default="/data2/member/SurfClip/data_surfclip/No_HCP_std.npy", type=str)

    parser.add_argument('--global_mean', default=None, type=str)
    parser.add_argument('--global_std', default=None, type=str)
    parser.add_argument('--save_freq', default=100, type=int)
    parser.add_argument('--recognition_index', default=3, type=int)
    parser.add_argument('--data-seed', default=1111, type=int)
    parser.add_argument('--freeze-depth', default=0, type=int)
    # parser.add_argument('--metadata', default='yfcc15m.pkl', type=str,
    #                 help='path to metadata file (see README for details)')
    parser.add_argument('--root', default='', type=str,
                        help='path to dataset root')
    parser.add_argument('--output-dir', default='./Output/No_HCP_SurfCLIP_finetune_10_mask0.75_weightmask_0.7_BioMedClip_freeze_test0.4_depth0_3333', type=str, help='path where to save, empty for no saving')

    # Data Augmentation
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.14, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    # Model
    parser.add_argument('--model', default='finetuneRegressionSurfClip_512', type=str)
    parser.add_argument('--mask-ratio', default=0.5, type=float)
    parser.add_argument('--ssl-mlp-dim', default=4096, type=int,
                        help='hidden dim of SimCLR mlp projection head')
    parser.add_argument('--ssl-emb-dim', default=256, type=int,
                        help='output embed dim of SimCLR mlp projection head')
    parser.add_argument('--ssl-scale', default=1.0, type=float,
                        help='loss scale for SimCLR objective')
    parser.add_argument('--ssl-temp', default=0.1, type=float,
                        help='softmax temperature for SimCLR objective')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    # Training
    parser.add_argument('--momentum-ema', default=0.996, type=float, help="""Base EMA
    parameter. The value is increased to 1 during training with cosine schedule.""")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=40, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--lr', default=0.001, type=float)
    # parser.add_argument('--base-lr', default=3e-3, type=float)
    parser.add_argument('--lr-start', default=0.0001, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-3, type=float,
                        help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.0, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate', action='store_true', help='eval only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--descriptions', default='training', type=str)
    parser.add_argument('--port', default=29500, help='port of master addr')
    # Loss
    parser.add_argument('--disable-norm-pix-loss', action='store_true',
                        help='disable normalization of pixel loss for reconstruction')
    parser.add_argument('--clip_loss_weight', default=1.0, type=float, help='weight of clip loss')
    # parser.add_argument('--ibot_patch_loss_weight', default=1.0, type=float, help='weight of ibot patch loss')
    parser.add_argument('--ibot_cls_loss_weight', default=1.0, type=float, help='weight of ibot classification loss') 
    parser.add_argument('--reconst_loss_weight', default=1.0, type=float, help='weight of reconstruction loss')
    parser.add_argument('--weight_mask', default=0.5, type=float, help='weight of reconstruction loss')

    return parser

def get_model(args):
    print("=> creating model: {}".format(args.model))
    model = getattr(SurfClip, args.model)(global_mean_path=args.global_mean,
                                          global_std_path=args.global_std,
                                          mask_ratio=args.mask_ratio,
                                          Pretrain = True) # note that args.mask_ratio is by default 0
    # model = getattr(CARZERO, args.model)()
    model.cuda(args.gpu)
    model = model.to(torch.float32)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200,find_unused_parameters=True)

    return model

def get_optim(args, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    return optimizer


def stratified_train_test_split(X, y, test_size=0.2, random_state=None):
    """
    分层训练测试分割，保持正负样本比例
    参数:
        X: 特征数据 (numpy 数组)
        y: 标签数据 (numpy 数组)
        test_size: 测试集比例 (默认 0.2)
        random_state: 随机种子 (默认 None)
    返回:
        X_train, X_test, y_train, y_test
    """
    # 设置随机种子
    rng = np.random.default_rng(random_state)
    mask = ~np.isnan(y)
    print(mask.shape)
    X= X[mask]
    y= y[mask]
    print("=> X.shape: {}".format(X.shape))
    print("=> y.shape: {}".format(y.shape))
    # 确保数据长度一致
    assert len(X) == len(y), "特征数据和标签数据长度不一致"

    # 初始化训练集和测试集索引列表
    train_indices = []
    test_indices = []

    class_indices = np.arange(0,len(X))
    num_class_samples = len(class_indices)

    # 计算该类别测试集的样本数量
    num_test_class = max(1, int(test_size * num_class_samples))
    print(num_test_class)
    # 随机打乱当前类别的索引
    rng.shuffle(class_indices)

    # 将当前类别样本划分为训练集和测试集
    class_test_indices = class_indices[:num_test_class]
    class_train_indices = class_indices[num_test_class:]

    # 添加到总索引列表
    test_indices.extend(class_test_indices)
    train_indices.extend(class_train_indices)

    # 将列表转换为numpy数组
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    print(train_indices.dtype)
    print(train_indices)
    # 最后再打乱一次避免类别顺序影响
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    # 提取训练集和测试集
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    print(train_indices.shape,test_indices.shape)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print(np.isnan(X_train).any())
    print(np.isnan(y_train).any())
    return X_train, X_test, y_train, y_test

def load_ckpt(args, model, optimizer, scaler):
    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            args.start_epoch = epoch
            result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(result)
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            args.best_acc = checkpoint['best_acc']
            print("=> loaded resume checkpoint '{}' (epoch {})"
                  .format(args.resume, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # auto-resume from latest checkpoint in output directory
        latest = os.path.join(args.output_dir, 'checkpoint.pt')
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu',weights_only=False)
            args.start_epoch = latest_checkpoint['epoch']
            model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            scaler.load_state_dict(latest_checkpoint['scaler'])
            args.best_acc = latest_checkpoint['best_acc']
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))
            
def get_loader(args, tokenizer):
    print("=> creating dataset")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=args.global_crops_scale, interpolation=3), # 3 is bicubic
        transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([Solarize()], p=0.2),
            transforms.ToTensor(),
            normalize
        ])

    train_dataset = get_dataset(train_transform, tokenizer, args)
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cwd, 'dataset_catalog.json')) as f:
        root = json.load(f)['imagenet']['path']
    #add val folder for imagenet 1k
    val_dataset = ImageFolder(os.path.join(root, 'val'), val_transform)

    # dist eval resamples data to pad uneven batch sizes
    # make sure num_samples = 0 mod num_gpus for exact acc
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)
    
    return train_loader, train_sampler, val_loader

def Surf_get_Dataloader(args):
    # transform = DataAugmentationForMAE_parcel(args)
    # print("Data Aug = %s" % str(transform))
    dataset_train = SurfaceFolder(args.root_path, None, image_path=args.train_image_filename,text_path=args.train_text_filename)
    dataset_val = SurfaceFolder(args.root_path, None,image_path=args.val_image_filename,text_path=args.val_text_filename)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)
    else:
        train_sampler = None
        val_sampler = None
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)
    return train_loader, train_sampler, val_loader

def Surf_get_finetune_Dataloader(args):
    # transform = DataAugmentationForMAE_parcel(args)
    # print("Data Aug = %s" % str(transform))
    data_path = os.path.join(args.root_path, args.surf_name)
    label_path = os.path.join(args.root_path, args.label_name)
    data = np.load(data_path)
    label = np.load(label_path).T[:,0].astype(float)
    if args.surf_name=="CHD.npy":
        label = np.load(label_path)[:, 0].astype(float)
    data = np.concatenate([data[:, :, :40962], data[:, :, 163842:163842 + 40962]], axis=2)
    if args.global_mean != None:
        mean = np.load(os.path.join(args.root_path, args.global_mean)).reshape([1,6,1])
        std = np.load(os.path.join(args.root_path, args.global_std)).reshape([1,6,1])
        data = (data - mean) / std

    X_train, X_test, y_train, y_test = stratified_train_test_split(data, label, random_state=args.data_seed)
    # print(sum(y_train == 0))
    # print(sum(y_train == 1))
    # print(sum(y_test == 0))
    # print(sum(y_test == 1))
    os.makedirs(args.save_path, exist_ok=True)
    np.save(os.path.join(args.save_path, "train_surf.npy"), X_train)
    np.save(os.path.join(args.save_path, "test_surf.npy"), X_test)
    np.save(os.path.join(args.save_path, "train_label.npy"), y_train)
    np.save(os.path.join(args.save_path, "test_label.npy"), y_test)

    # config the path name
    args.train_image_filename = os.path.join(args.save_path, "train_surf.npy")
    args.train_label_filename = os.path.join(args.save_path, "train_label.npy")
    args.val_image_filename = os.path.join(args.save_path, "test_surf.npy")
    args.val_label_filename = os.path.join(args.save_path, "test_label.npy")

    dataset_train = Finetune_SurfaceFolder(args.root_path, None, image_path=args.train_image_filename,label_path=args.train_label_filename)
    dataset_val = Finetune_SurfaceFolder(args.root_path, None,image_path=args.val_image_filename,label_path=args.val_label_filename)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)
    else:
        train_sampler = None
        val_sampler = None
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)
    return train_loader, train_sampler, val_loader

def load_state_dict_1(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

def load_pretrained_weights(args, model):
    if args.finetune == "None":
        return model
    if args.finetune.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.finetune, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)


    print("Load ckpt from %s" % args.finetune)
    checkpoint_model = None
    new_state_dict = OrderedDict()
    for model_key in checkpoint["state_dict"].keys():
        if model_key.startswith('module.tokenizer'):
            new_state_dict[model_key[:]] = checkpoint["state_dict"][model_key]
        elif model_key.startswith('module.visual.'):
            new_state_dict[model_key[:]] = checkpoint["state_dict"][model_key]
        else:
            continue

    checkpoint_model = new_state_dict
    load_state_dict_1(model, checkpoint_model)
    return model


def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    计算二分类模型的评估指标

    参数:
    y_true -- 真实标签 (numpy数组)
    y_pred_proba -- 预测概率 (numpy数组)
    threshold -- 将概率转换为二元预测的阈值 (默认0.5)

    返回:
    包含所有评估指标的字典
    """
    # 使用阈值将概率转换为二元预测
    y_pred = (y_pred_proba >= threshold).astype(int)[:,1]
    y_pred_proba = y_pred_proba[:,1]
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 计算各项指标
    sensitivity = recall_score(y_true, y_pred)  # 敏感性/召回率
    specificity = tn / (tn + fp)  # 特异性
    precision = precision_score(y_true, y_pred)  # 精确率
    f1 = f1_score(y_true, y_pred)  # F1分数

    # 计算AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    # 计算ROC曲线和PR曲线
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)

    return {
        'Confusion Matrix': [[tn, fp], [fn, tp]],
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1_Score': f1,
        'ROC_AUC': roc_auc,
        'PR AUC': pr_auc,
        'Threshold': threshold,
        'ROC Curve': (fpr, tpr),
        'PR Curve': (recall_curve, precision_curve)
    }

def freeze_visual_layers(model, freeze_depth=0):
    visual_model = model.module.visual.blocks
    tokenizer = getattr(model.module, "tokenizer", None)
    for i, (name, child) in enumerate(model.module.visual.named_children()):
        for param in child.parameters():
            param.requires_grad = False

    for i, child in enumerate(visual_model):
        if i >= freeze_depth:
            # 冻结浅层
            for param in child.parameters():
                param.requires_grad = False
    if tokenizer!= None:
        for i, (name, child) in enumerate(tokenizer.named_children()):
            for param in child.parameters():
                param.requires_grad = False
    return model

def freeze_visual_layers_car(model, freeze_depth=10):
    visual_model = model.module.img_encoder
    tokenizer = getattr(model.module, "tokenizer", None)
    for i, (name, child) in enumerate(model.module.img_encoder.named_children()):
        for param in child.parameters():
            param.requires_grad = False
    if tokenizer != None:
        for i, (name, child) in enumerate(tokenizer.named_children()):
            for param in child.parameters():
                param.requires_grad = False
    return model

def main(args):
    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version()
    print(f"CUDA Version: {cuda_version}")
    print(f"cuDNN Version: {cudnn_version}")

    # 初始化分布式环境
    dist.init_process_group(
        backend="nccl",
        init_method="env://"  # 从环境变量自动读取 MASTER_ADDR 和 MASTER_PORT
    )

    # 获取分布式训练参数
    rank = dist.get_rank()  # 全局进程编号
    world_size = dist.get_world_size()  # 总进程数
    local_rank = int(os.environ["LOCAL_RANK"])  # 从环境变量获取本地 rank

    # 打印分布式信息
    print(f"Global Rank: {rank}")
    print(f"Local Rank: {local_rank}")
    print(f"World Size: {world_size}")

    # 设置当前设备
    torch.cuda.set_device(local_rank)
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    # addr = subprocess.getoutput(
    #     f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if args.port is not None:
        os.environ['MASTER_PORT'] = str(args.port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    # os.environ['WORLD_SIZE'] = str(ntasks)
    # # os.environ['RANK'] = str(proc_id)
    #
    # #utils.init_distributed_mode(args)
    # ##---------------------------------
    # #os.environ['MASTER_ADDR'] = addr
    # os.environ['WORLD_SIZE'] = str(ntasks)
    # os.environ['RANK'] = str(proc_id)
    print('| distributed init (rank {}): {}'.format(
        int(os.environ["RANK"]), args.dist_url), flush=True)
    args.distributed = True
    # dist.init_process_group(backend='nccl')
    torch.distributed.barrier()
    utils.setup_for_distributed(args.rank == 0)
    ##---------------------------------
    cudnn.benchmark = True


    # args.patch_size = int(re.search(r'\d+', args.model).group(0)) # extract patch size from model name

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.MSELoss()
    temp = args.output_dir

    
    # Data loading 
    # tokenizer = SimpleTokenizer()
    # tokenizer = clip.tokenize()
    # train_loader, train_sampler, val_loader = get_loader(args, tokenizer)

    train_loader, train_sampler, val_loader = Surf_get_finetune_Dataloader(args)

    # if args.evaluate:
    #     zero_stats = validate_zeroshot(val_loader, model, tokenizer, args)
    #     if utils.is_main_process():
    #         with open(os.path.join(args.output_dir, 'eval_log.txt'), 'a') as f:
    #             f.write(json.dumps(zero_stats) + '\n')
    #     return

    lr_schedule = utils.cosine_scheduler(args.lr, args.lr_end, args.epochs,
        len(train_loader) // args.update_freq, warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start)

    # momentum_schedule = utils.cosine_scheduler(args.momentum_ema, 1, args.epochs, len(train_loader), 0)

    if utils.is_main_process() and args.wandb:
        wandb_id = os.path.split(args.output_dir)[-1]
        wandb.init(project='DetailCLIP', id=wandb_id, config=args, resume='resume')

    print(args)

    print("=> beginning training")
    best_mse = []
    best_mae = []
    for iter in range(1):
        args.best_mse = 9999
        args.best_mae = 9999

        args.output_dir = temp+ f"_{iter}"
        # create model
        model = get_model(args)
        # load model distributed
        model = load_pretrained_weights(args, model)
        optimizer = get_optim(args, model)
        scaler = amp.GradScaler(enabled=not args.disable_amp)

        load_ckpt(args, model, optimizer, scaler)
        model = freeze_visual_layers(model, freeze_depth=args.freeze_depth)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            # train for one epoch
            train_stats = train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args)

            if (epoch + 1) % args.eval_freq != 0:
                continue

            val_stats = validate_zeroshot(val_loader, model, args)
            mse = val_stats['mse']
            mae = val_stats['mae']
            is_best =  mae < args.best_mae
            if is_best:
                args.best_mse = mse
                args.best_mae = mae

            print("=> saving checkpoint")
            utils.save_on_master_regression({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_mae': args.best_mae,
                    'best_mse': args.best_mse,
                    'args': args,
                }, is_best, args.output_dir,epoch+1,args.save_freq,args.epochs)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch}

            if utils.is_main_process():
                if args.wandb:
                    wandb.log(log_stats)
                with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                    f.write(json.dumps(log_stats) + '\n')
        best_mse.append(args.best_mse)
        best_mae.append(args.best_mae)
        np.save(args.output_dir + '/best_mse.npy', best_mse)
        np.save(args.output_dir + '/best_mae.npy', best_mae)
    print(best_mse)
    print(best_mae)


def train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.3f')
    metric_names = get_finetuning_surfclip_names()
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it]
        
        online_inputs = [inputs[0],inputs[1]]
        online_inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in online_inputs]

        # m = momentum_schedule[it]  # momentum parameter
        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(online_inputs[0])
            loss = criterion(outputs["cls"].reshape([-1]), online_inputs[1])
            loss_dict = {'loss': loss}
            loss /= args.update_freq

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        scaler.scale(loss).backward()

        if (data_iter + 1) % args.update_freq != 0:
            continue

        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)

        end = time.time()
        
        mem.update(torch.cuda.max_memory_allocated() / 1e9)
        if optim_iter % args.print_freq == 0:
            if utils.is_main_process() and args.wandb:
                wandb.log({**{k: v.item() for k, v in loss_dict.items()},
                        'scaler': scaler.get_scale(),
                        })
            progress.display(optim_iter)

    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr']}


def validate_zeroshot(val_loader, model, args, ema=False):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('mse@1', ':6.2f')
    top2 = AverageMeter('mae@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top2],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        image_features = []
        targets = []
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # encode images
            model = utils.get_model(model)
            prediction = model(images)
            image_features.append(prediction['cls'])
            targets.append(target)
        image_features = torch.cat(image_features,dim=0)
        targets = torch.cat(targets,dim=0)
        mse = torch.mean(torch.square(image_features-targets))
        mae = torch.mean(torch.abs(image_features - targets))
        top1.update(mse.item(), image_features.size(0))
        top2.update(mae.item(), image_features.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    progress.synchronize()
    print('0-shot * mse@1 {top1.avg:.3f}'
          .format(top1=top1))
    print('0-shot * mae@1 {top2.avg:.3f}'
          .format(top2=top2))
    return {'mse': top1.avg, 'mae': top2.avg}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DetailCLIP training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)
