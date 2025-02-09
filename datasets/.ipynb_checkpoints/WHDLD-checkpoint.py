import os
import os.path as osp
import numpy as np
from PIL import Image
import cv2
from torch.utils import data
import logging
from config import cfg

num_classes = 6     # 定义了数据集中的类别数
ignore_label = 255  # 设置需要忽略的标签值
root = cfg.DATASET.WHDLD_DIR   # 设置数据集的根目录 '/home/featurize/data/WHDLD-xzh'

# 标签与训练 ID 之间的映射
label2trainid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
# label2trainid = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
# label2trainid = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}

# ID 与类别名称之间的映射
id2cat = {0: 'buildings', 1: 'road', 2: 'pavement',
          3: 'vegetation', 4: 'bare soil', 5: 'water'}
# id2cat = {1: 'buildings', 2: 'road', 3: 'pavement',
#           4: 'vegetation', 5: 'bare soil', 6: 'water'}

# 定义了一个调色板，用于将标签图像转换为彩色图像
palette = [255, 0, 0, 255, 255, 0, 192, 192, 0, 0, 255, 0, 128, 128, 128, 0, 0, 255]
# palette = [0, 0, 255, 0, 255, 255, 0, 192, 192, 0, 255, 0, 128, 128, 128, 255 ,0, 0]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

# colorize_label 函数接受一个 label 作为输入，并将其转换为彩色图像
def colorize_label(label):
    new_label = Image.fromarray(label.astype(np.int8)).convert('P')
    new_label.putpalette(palette)
    return new_label

# make_dataset 函数用于创建一个包含图像和标签对的数据集
def make_dataset(quality, mode):
    all_tokens = []

    assert quality == 'semantic'    # 数据集包含的是语义标注
    assert mode in ['train', 'val', 'test']

    image_path = osp.join(root, mode, 'images')
    label_path = osp.join(root, mode, 'labels0319')

    image_tokens = os.listdir(image_path)
    image_tokens.sort()
    label_tokens = [c_token.replace('.jpg', '.png') for c_token in image_tokens]

    for image_token, label_token in zip(image_tokens, label_tokens):
        token = (osp.join(image_path, image_token), osp.join(label_path, label_token))
        all_tokens.append(token)
    # 使用 logging.info 函数记录有关数据集的信息，包括数据集中图像的总数和当前的模式。
    # logging.info(f'WHDLD-xzh has a total of {len(all_tokens)} images in {mode} phase')
    logging.info(f'WHDLD-{mode}: {len(all_tokens)} images')
    return all_tokens

# 定义类用于加载和预处理WHDLD数据集的图像和掩码
class WHDLDDataset(data.Dataset):
    # 类的构造函数，用于初始化类的属性
    def __init__(self, quality, mode, maxSkip=0, joint_transform_list=None,
                 transform=None, label_transform=None, dump_images=False,
                 class_uniform_pct=None, class_uniform_title=0, test=False,
                 cv_split=None, scf=None, hardnm=0, edge_map=False, thicky=8):

        self.quality = quality      # 数据集的质量 'semantic'
        self.mode = mode    # 数据集的模式，可以是 'train'、'val'、'test'
        self.maxSkip = maxSkip  # 一个可选参数，用于指定在数据集中跳过多少个图像
        self.joint_transform_list = joint_transform_list    # 图像和掩码的变换
        self.transform = transform  # 仅应用于图像的变换
        self.label_transform = label_transform    # 仅应用于标签的变换
        self.dump_images = dump_images  # 如果设置为 True，则会在一个特定的目录中保存图像和掩码的副本
        self.class_uniform_pct = class_uniform_pct  # 指定类别均匀分布的百分比
        self.class_uniform_title = class_uniform_title  # 指定类别均匀分布的标题
        if cv_split:    # 用于指定交叉验证的分割
            self.cv_split = cv_split
            assert cv_split < cfg.DATASET.CV_SPLITS
        self.scf = scf
        self.hardnm = hardnm
        self.edge_map = edge_map    # 如果设置为 True，则返回图像、掩码、边界和主体区域

        self.data_tokens = make_dataset(quality, mode)
        self.thicky = thicky    # 用于边界提取的厚度

        assert len(self.data_tokens), 'Found 0 images please check the dataset'

    def __getitem__(self, index):

        token = self.data_tokens[index]     # 根据[索引]返回(图像路径:标签路径)
        image_path, label_path = token

        image, label = Image.open(image_path).convert('RGB'), Image.open(label_path)
        image_name = osp.splitext(osp.basename(image_path))[0]  # 提取图像名，并去掉扩展名

        # 图像和标签一起变换
        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):     # enumerate 函数用于同时获取列表的索引和元素
                image, label = xform(image, label)   # 评估时用
                # 下面两行训练时用
                # image = xform(image)
                # label = xform(label)

        # 检查变换是否正确。dump_images若为True，则保存image和label
        if self.dump_images:
            outdir = '/lb-img/dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            out_img_fn = os.path.join(outdir, image_name + '.png')
            out_lb_fn = os.path.join(outdir, image_name + '_label.png')
            label_img = colorize_label(np.array(label))
            image.save(out_img_fn)
            label_img.save(out_lb_fn)

        # 对图像进行单独变换
        if self.transform is not None:
            image = self.transform(image)

        # 对标签进行单独变换
        if self.label_transform is not None:
            label = self.label_transform(label)


        if self.edge_map:
            boundary = self.get_boundary(label, thicky=self.thicky)
            body = self.get_body(label, boundary)
            return image, label, body, boundary, image_name


        return image, label, image_name

    def __len__(self):
        return len(self.data_tokens)

    def build_epoch(self):
        pass

    # 用于从掩码图像中提取边界
    @staticmethod
    def get_boundary(label, thicky=8):
        tmp = label.data.numpy().astype('uint8')
        contour, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boundary = np.zeros_like(tmp)
        boundary = cv2.drawContours(boundary, contour, -1, 1, thicky)
        boundary = boundary.astype(np.float)
        return boundary

    # 用于从掩码图像中提取主体区域
    @staticmethod
    def get_body(label, edge):
        edge_valid = edge == 1
        body = label.clone()
        body[edge_valid] = ignore_label
        return body












