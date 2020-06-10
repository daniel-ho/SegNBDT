import os
import json

import cv2
import numpy as np
from PIL import Image

import torch

from .base_dataset import BaseDataset

class ADE20K(BaseDataset):
    def __init__(self, 
                 root,
                 list_path,
                 num_samples=None, 
                 num_classes=150,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=512,
                 crop_size=(512, 512),
                 center_crop_test=False,
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(ADE20K, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.center_crop_test = center_crop_test

        self.img_list = [json.loads(x.rstrip()) for x in open(root+list_path, 'r')]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        for item in self.img_list:
            image_path = item['fpath_img'].replace('ADEChallengeData2016', 'ade20k')
            label_path = item['fpath_segm'].replace('ADEChallengeData2016', 'ade20k')
            name = os.path.splitext(os.path.basename(image_path))[0]
            files.append({
                'img': image_path,
                'label': label_path,
                'name': name,
                })
        return files

    def resize_image_label(self, image, label, size):
        scale = size/min(image.shape[0], image.shape[1])
        image = cv2.resize(image, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) 
        label = cv2.resize(label, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return image, label

    def convert_label(self, label):
        # Convert labels to -1 to 149
        return np.array(label).astype('int32') - 1

    def __getitem__(self, index):
        item = self.files[index]
        name = item['name']
        image = cv2.imread(os.path.join(self.root, item['img']), cv2.IMREAD_COLOR)
        size = image.shape
        label = cv2.imread(os.path.join(self.root, item['label']), cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        if 'validation' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2,0,1))
            label = self.label_transform(label)
        else:
            image, label = self.resize_image_label(image, label, self.base_size)
            image, label = self.gen_sample(image, label, 
                                    self.multi_scale, self.flip, 
                                    self.center_crop_test)

        return image.copy(), label.copy(), np.array(size), name

