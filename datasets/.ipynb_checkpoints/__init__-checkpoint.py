"""
__init__.py 文件是 datasets 包的入口点，它定义了数据集的设置和加载流程。
"""
# from datasets import iSAID
from datasets import potsdam
from datasets import Vaihingen
from datasets import WHDLD

import torchvision.transforms as standard_transforms
# from torchvision import transforms
import transforms.joint_transforms as joint_transforms
import transforms.transforms as extended_transforms
from torch.utils.data import DataLoader


def setup_loaders(args):
    if args.dataset == 'Potsdam':
        args.dataset_cls = potsdam
        args.train_batch_size = args.bs_mult * args.ngpu
        if args.bs_mult_val > 0:
            args.val_batch_size = args.bs_mult_val * args.ngpu
        else:
            args.val_batch_size = args.bs_mult * args.ngpu

    elif args.dataset == 'Vaihingen':
        args.dataset_cls = Vaihingen
        args.train_batch_size = args.bs_mult * args.ngpu
        if args.bs_mult_val > 0:
            args.val_batch_size = args.bs_mult_val * args.ngpu
        else:
            args.val_batch_size = args.bs_mult * args.ngpu
    elif args.dataset == 'WHDLD':
        args.dataset_cls = WHDLD
        args.train_batch_size = args.bs_mult * args.ngpu
        if args.bs_mult_val > 0:
            args.val_batch_size = args.bs_mult_val * args.ngpu
        else:
            args.val_batch_size = args.bs_mult * args.ngpu
    else:
        raise Exception('Dataset {} is not supported'.format(args.dataset))

    """
    image和label共同的数据增强
    """

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # 大小、方向、角度的变换
    # train_joint_transform_list = [transforms.Resize(args.crop_size),
    #                               transforms.RandomHorizontalFlip(),
    #                               transforms.RandomRotation(5)]
    # 图像几何变换
    train_joint_transform_list = [
        joint_transforms.RandomSizeAndCrop(args.crop_size,
                                        False,
                                        pre_size=args.pre_size,
                                        scale_min=args.scale_min,
                                        scale_max=args.scale_max,
                                        ignore_index=args.dataset_cls.ignore_label),
        joint_transforms.Resize(args.crop_size),
        joint_transforms.RandomHorizontallyFlip()]
    
    if args.dataset == 'WHDLD':
        train_joint_transform_list = [
            joint_transforms.Resize(args.crop_size),
            joint_transforms.RandomHorizontallyFlip(),
            joint_transforms.RandomVerticalFlip(),
            joint_transforms.RandomRotateThreeDegree()]
        val_joint_transform_list = [
            joint_transforms.Resize(args.crop_size)
        ]
    # if args.dataset == 'WHDLD' and args.with_aug:
    #     train_joint_transform_list = [
    #         joint_transforms.RandomSizeAndCrop(args.crop_size,
    #                                            False,
    #                                            pre_size=args.pre_size,
    #                                            scale_min=args.scale_min,
    #                                            scale_max=args.scale_max,
    #                                            ignore_index=args.dataset_cls.ignore_label),
    #         joint_transforms.RandomHorizontallyFlip(),
    #         joint_transforms.RandomRotateThreeDegree()]
    #     val_joint_transform_list = [
    #         joint_transforms.Resize(args.crop_size)
    #     ]
    if args.dataset == 'Potsdam' and not args.with_aug:
        train_joint_transform_list = [
            joint_transforms.Resize(args.crop_size),
            joint_transforms.RandomHorizontallyFlip(),
            joint_transforms.RandomRotateThreeDegree()]
        val_joint_transform_list = [
            joint_transforms.Resize(args.crop_size)
        ]
    if args.dataset == 'Potsdam' and args.with_aug:
        train_joint_transform_list = [
            joint_transforms.RandomSizeAndCrop(args.crop_size,
                                               False,
                                               pre_size=args.pre_size,
                                               scale_min=args.scale_min,
                                               scale_max=args.scale_max,
                                               ignore_index=args.dataset_cls.ignore_label),
            joint_transforms.RandomHorizontallyFlip(),
            joint_transforms.RandomRotateThreeDegree()]
        val_joint_transform_list = [
            joint_transforms.Resize(args.crop_size)
        ]
    if args.dataset == 'Vaihingen' and not args.with_aug:
        train_joint_transform_list = [
            joint_transforms.Resize(args.crop_size),
            joint_transforms.RandomHorizontallyFlip(),
            joint_transforms.RandomRotateThreeDegree()]
        val_joint_transform_list = [
            joint_transforms.Resize(args.crop_size)
        ]
    if args.dataset == 'Vaihingen' and args.with_aug:
        train_joint_transform_list = [
            joint_transforms.RandomSizeAndCrop(args.crop_size,
                                               False,
                                               pre_size=args.pre_size,
                                               scale_min=args.scale_min,
                                               scale_max=args.scale_max,
                                               ignore_index=args.dataset_cls.ignore_label),
            joint_transforms.RandomHorizontallyFlip(),
            joint_transforms.RandomRotateThreeDegree()]
        val_joint_transform_list = [
            joint_transforms.Resize(args.crop_size)
        ]

    # val_joint_transform_list = [transforms.Resize(args.crop_size)]
    # 图像外观变换
    train_input_transform = []
    # if args.color_aug:  # 颜色抖动
    #     train_input_transform += [transforms.ColorJitter(brightness=args.color_aug,
    #                                                      contrast=args.color_aug,
    #                                                      saturation=args.color_aug,
    #                                                      hue=args.color_aug)]
    if args.color_aug:
        train_input_transform += [extended_transforms.ColorJitter(
            brightness=args.color_aug,
            contrast=args.color_aug,
            saturation=args.color_aug,
            hue=args.color_aug)]

    if args.bblur:  # 双边模糊
        train_input_transform += [extended_transforms.RandomBilateralBlur()]
    elif args.gblur:    # 高斯模糊
        train_input_transform += [extended_transforms.RandomGaussianBlur()]
    else:
        pass

    train_input_transform += [standard_transforms.ToTensor(),
                              standard_transforms.Normalize(*mean_std)]
    train_input_transform = standard_transforms.Compose(train_input_transform)
    # 将label文件转换为tensor类型
    target_transform = extended_transforms.MaskToTensor()

    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    ## relax the segmentation border
    if args.jointwtborder:  # 是否启用边界加权
        target_train_transform = extended_transforms.RelaxedBoundaryLossToTensor(args.dataset_cls.ignore_label,
            args.dataset_cls.num_classes)
    else:
        target_train_transform = extended_transforms.MaskToTensor()

    edge_map = args.joint_edge_loss_pfnet  # 是否使用联合边缘分割损失

    if args.dataset == 'Potsdam':
        train_set = args.dataset_cls.PotsdamDataset(
            'semantic', 'train', args.maxSkip,
            joint_transform_list=train_joint_transform_list,
            transform=train_input_transform,
            label_transform=target_train_transform,
            dump_images=args.dump_augmentation_images,
            class_uniform_pct=args.class_uniform_pct,
            class_uniform_title=args.class_uniform_tile,
            test=args.test_mode,
            cv_split=args.cv,
            scf=args.scf,
            hardnm=args.hardnm,
            edge_map=edge_map,
            thicky=args.thicky)
        val_set = args.dataset_cls.PotsdamDataset(
            'semantic', 'test', 0,
            joint_transform_list=val_joint_transform_list,
            transform=val_input_transform,
            label_transform=target_transform,
            test=False,
            cv_split=args.cv,
            scf=None)
    elif args.dataset == 'Vaihingen':
        train_set = args.dataset_cls.VAIHINGENDataset(
            'semantic', 'train', args.maxSkip,
            joint_transform_list=train_joint_transform_list,
            transform=train_input_transform,
            label_transform=target_train_transform,
            dump_images=args.dump_augmentation_images,
            class_uniform_pct=args.class_uniform_pct,
            class_uniform_title=args.class_uniform_tile,
            test=args.test_mode,
            cv_split=args.cv,
            scf=args.scf,
            hardnm=args.hardnm,
            edge_map=edge_map,
            thicky=args.thicky)
        val_set = args.dataset_cls.VAIHINGENDataset(
            'semantic', 'test', 0,
            joint_transform_list=val_joint_transform_list,
            transform=val_input_transform,
            label_transform=target_transform,
            test=False,
            cv_split=args.cv,
            scf=None)
    elif args.dataset == 'WHDLD':
        train_set = args.dataset_cls.WHDLDDataset(
            'semantic', 'train', args.maxSkip,
            joint_transform_list=train_joint_transform_list,
            transform=train_input_transform,
            label_transform=target_train_transform,
            dump_images=args.dump_augmentation_images,
            class_uniform_pct=args.class_uniform_pct,
            class_uniform_title=args.class_uniform_tile,
            test=args.test_mode,
            cv_split=args.cv,
            scf=args.scf,
            hardnm=args.hardnm,
            edge_map=edge_map,
            thicky=args.thicky)
        val_set = args.dataset_cls.WHDLDDataset(
            'semantic', 'test', 0,
            joint_transform_list=val_joint_transform_list,
            transform=val_input_transform,
            label_transform=target_transform,
            test=False,
            cv_split=args.cv,
            scf=None)
        
    elif args.dataset == 'null_loader':  # 检查命令行参数 args.dataset 是否等于 'null_loader'
        train_set = args.dataset_cls.null_loader(args.crop_size)
        val_set = args.dataset_cls.null_loader(args.crop_size)
    else:   # 如果不是 null_loader，则抛出异常
        raise Exception('Dataset {} is not supported'.format(args.dataset))
    
    if args.apex:   # 如果为 True，则表示用户使用了 Nvidia Apex 库进行训练
        # 设置分布式采样器，用于在多 GPU 训练中分配样本。
        from datasets.sampler import DistributedSampler
        train_sampler = DistributedSampler(train_set, pad=True, permutation=True, consecutive_sample=False)
        val_sampler = DistributedSampler(val_set, pad=False, permutation=False, consecutive_sample=False)

    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size,
                              num_workers=args.num_workers, shuffle=(train_sampler is None), drop_last=True, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size,
                            num_workers=args.num_workers // 2 , shuffle=False, drop_last=False, sampler=val_sampler)

    return train_loader, val_loader, train_set