import os
import time
import copy
from collections import defaultdict
import torch
import shutil
import pandas as pd
from skimage import io, transform
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm as tqdm

from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
import cv2

from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch.nn.functional as F
from PIL import Image
from torch import nn
import zipfile

import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TRAIN_PATH = 'data/stage1_train/'


def get_train_transform():
    return A.Compose(
        [
            A.Resize(256, 256), #リサイズ
            # (img - mean * 255) / (std * 255)
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.HorizontalFlip(p=0.25), #水平方向のフリップ
            A.VerticalFlip(p=0.25), #垂直方向のフリップ
            ToTensorV2()
        ]
    )

class LoadDataSet(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.folders = os.listdir(path)
        self.transform = transform

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        image_folder = os.path.join(
            self.path, self.folders[idx], "images/"
        )
        mask_folder = os.path.join(
            self.path, self.folders[idx], "masks/"
        )
        image_path = os.path.join(
            image_folder, os.listdir(image_folder)[0]
        )
        # 画像データの取得
        img = io.imread(image_path)[:, :, :3].astype("float32")
        img = transform.resize(img, (256, 256))
        mask = self.get_mask(mask_folder, 256, 256).astype("float32")

        augmented = self.transform(image=img, mask=mask)
        img = augmented["image"] #TODO: ["image"]はどうやって決まっている？
        mask = augmented["mask"] #TODO: ["mask"]はどうやって決まっている？
        mask = mask.permute(2, 0, 1)  #TODO: mask[0]はどういう意味？->いらない
        return (img, mask)

    def get_mask(self, mask_folder, IMG_HEIGHT, IMG_WIDTH):
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)
        for mask_ in os.listdir(mask_folder):
            mask_ = io.imread(
                os.path.join(mask_folder, mask_)
            )
            mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
            mask_ = np.expand_dims(mask_, axis=-1) # (height, width) -> (height, width, 1)
            mask = np.maximum(mask, mask_) # ベースのmaskとmask_を比較

        return mask


train_dataset = LoadDataSet(TRAIN_PATH, transform=get_train_transform())

image, mask = train_dataset.__getitem__(0)

def format_image(img):
    img = np.array(np.transpose(img, (1, 2, 0)))
    mean = np.array((0.485, 0.456, 0.406))
    std = np.array((0.229, 0.224, 0.225))
    print("before\n", img)
    img = std * img + mean
    print("after\n", img)
    img = img * 255
    img = img.astype(np.uint8)
    return img

def format_mask(mask):
    mask = np.squeeze(np.transpose(mask, (1, 2, 0)))
    return mask

def visualize_dataset(n_images, predict=None):
    images = random.sample(range(0, 670), n_images)
    figure, ax = plt.subplots(nrows=len(images), ncols=2, figsize=(5, 8))
    for i in range(0, len(images)):
        img_no = images[i]
        image, mask = train_dataset.__getitem__(img_no)
        image = format_image(image)
        mask = format_mask(mask)
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest", cmap="gray")
        ax[i, 0].set_title("Input Image")
        ax[i, 1].set_title("Label Mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
    plt.tight_layout()
    plt.show()


split_ratio = 0.25
train_size = int(np.round(train_dataset.__len__() * (1 - split_ratio), 0))