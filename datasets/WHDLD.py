import os
import os.path as osp
import numpy as np
from PIL import Image
import cv2
from torch.utils import data
import logging
from config import cfg

num_classes = 6
ignore_label = 255
root = cfg.DATASET.WHDLD_DIR

label2trainid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

id2cat = {0: 'buildings', 1: 'road', 2: 'pavement',
          3: 'vegetation', 4: 'bare soil', 5: 'water'}

palette = [255, 0, 0, 255, 255, 0, 192, 192, 0, 0, 255, 0, 128, 128, 128, 0, 0, 255]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_label(label):
    new_label = Image.fromarray(label.astype(np.int8)).convert('P')
    new_label.putpalette(palette)
    return new_label

def make_dataset(quality, mode):
    all_tokens = []

    assert quality == 'semantic'
    assert mode in ['train', 'val', 'test']

    image_path = osp.join(root, mode, 'images')
    label_path = osp.join(root, mode, 'labels0319')

    image_tokens = os.listdir(image_path)
    image_tokens.sort()
    label_tokens = [c_token.replace('.jpg', '.png') for c_token in image_tokens]

    for image_token, label_token in zip(image_tokens, label_tokens):
        token = (osp.join(image_path, image_token), osp.join(label_path, label_token))
        all_tokens.append(token)

    logging.info(f'WHDLD-{mode}: {len(all_tokens)} images')
    return all_tokens


class WHDLDDataset(data.Dataset):
    def __init__(self, quality, mode, maxSkip=0, joint_transform_list=None,
                 transform=None, label_transform=None, dump_images=False,
                 class_uniform_pct=None, class_uniform_title=0, test=False,
                 cv_split=None, scf=None, hardnm=0, edge_map=False, thicky=8):

        self.quality = quality
        self.mode = mode
        self.maxSkip = maxSkip
        self.joint_transform_list = joint_transform_list
        self.transform = transform
        self.label_transform = label_transform
        self.dump_images = dump_images
        self.class_uniform_pct = class_uniform_pct
        self.class_uniform_title = class_uniform_title
        if cv_split:
            self.cv_split = cv_split
            assert cv_split < cfg.DATASET.CV_SPLITS
        self.scf = scf
        self.hardnm = hardnm
        self.edge_map = edge_map

        self.data_tokens = make_dataset(quality, mode)
        self.thicky = thicky

        assert len(self.data_tokens), 'Found 0 images please check the dataset'

    def __getitem__(self, index):

        token = self.data_tokens[index]
        image_path, label_path = token

        image, label = Image.open(image_path).convert('RGB'), Image.open(label_path)
        image_name = osp.splitext(osp.basename(image_path))[0]

        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                image, label = xform(image, label)

        if self.dump_images:
            outdir = '/lb-img/dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            out_img_fn = os.path.join(outdir, image_name + '.png')
            out_lb_fn = os.path.join(outdir, image_name + '_label.png')
            label_img = colorize_label(np.array(label))
            image.save(out_img_fn)
            label_img.save(out_lb_fn)

        if self.transform is not None:
            image = self.transform(image)

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

    @staticmethod
    def get_boundary(label, thicky=8):
        tmp = label.data.numpy().astype('uint8')
        contour, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boundary = np.zeros_like(tmp)
        boundary = cv2.drawContours(boundary, contour, -1, 1, thicky)
        boundary = boundary.astype(np.float)
        return boundary

    @staticmethod
    def get_body(label, edge):
        edge_valid = edge == 1
        body = label.clone()
        body[edge_valid] = ignore_label
        return body

