import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms.functional as F

import os
import pandas as pd
import numpy as np
import random

MEAN = (0.5, 0.5, 0.5)
VAR = (0.5, 0.5, 0.5)


def parse_list(root):
    inputs = []
    for name in os.listdir(root):
        if os.path.isdir(os.path.join(root, name)):
            for file in os.listdir(os.path.join(root, name)):
                inputs.append(os.path.join(name, file.split(sep='.')[0]))
    return inputs


def get_index(name, images_root='images', text_root='text', miss=None):
    if miss is not None:
        path_miss = os.path.join(text_root, miss + '.txt')
        path_text = os.path.join(text_root, name + '.txt')
        path_image = os.path.join(images_root, name + '.jpg')
        return {'text': path_text, 'image': path_image, 'miss': path_miss}
    else:
        path_text = os.path.join(text_root, name + '.txt')
        path_image = os.path.join(images_root, name + '.jpg')
        return {'text': path_text, 'image': path_image}


class TextDatasetCreator(torch.utils.data.Dataset):
    def __init__(self, text_root='text', images_root='images'):
        super().__init__()
        self.text_root = text_root
        self.images_root = images_root
        self.paths = parse_list(text_root)

        self.dset_size = len(self.paths)

    def __len__(self):
        return self.dset_size

    def __getitem__(self, item):
        item_path = self.paths[item % self.dset_size]
        paths = get_index(item_path, self.images_root, self.text_root)

        text_path = paths['text']

        texts = []
        with open(text_path) as f:
            text = f.readlines()
            text = [line.strip() for line in text]

        return text


class TextToImageDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, text_root='text', images_root='images', bbox_path='bounding_boxes.txt', paths='images.txt'):
        super().__init__()
        self.text_root = text_root
        self.images_root = images_root
        self.paths = parse_list(text_root)

        self.dset_size = len(self.paths)
        self.img_size = (cfg.load_size, cfg.load_size)
        self.load_size = cfg.load_size
        self.scales = [int(cfg.load_size / (2 ** i)) for i in range(3)]
        self.scales.reverse()

        self.df_bounding_boxes = pd.read_csv(bbox_path, delim_whitespace=True, header=None).astype(int)
        self.df_filenames = pd.read_csv(paths, delim_whitespace=True, header=None)
        filenames = self.df_filenames[1].tolist()

        self.filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            bbox = self.df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            self.filename_bbox[key] = bbox


    def __len__(self):
        return self.dset_size

    def __getitem__(self, item):
        item_path = self.paths[item % self.dset_size]
        paths = get_index(item_path, self.images_root, self.text_root)

        text_path = paths['text']
        image_path = paths['image']

        texts = []
        with open(text_path) as f:
            text = f.readlines()
            text = [line.strip() for line in text]

        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        bbox = self.filename_bbox[item_path]

        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        image = image.crop([x1, y1, x2, y2])

        scales = {}
        for scale in self.scales:
            scale_size = scale
            scales[str(scale)] = F.resize(image, (scale_size, scale_size))
            scales[str(scale)] = F.to_tensor(scales[str(scale)])
            scales[str(scale)] = F.normalize(scales[str(scale)], MEAN, VAR, inplace=True)

        return {'text': text, 'images': scales}


class MismatchTextToImageDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, text_root='text', images_root='images', bbox_path='bounding_boxes.txt', paths='images.txt'):
        super().__init__()
        self.text_root = text_root
        self.images_root = images_root
        self.paths = parse_list(text_root)

        self.dset_size = len(self.paths)
        self.img_size = (cfg.load_size, cfg.load_size)
        self.load_size = cfg.load_size
        self.scales = [int(cfg.load_size / (2 ** i)) for i in range(3)]
        self.scales.reverse()

        self.df_bounding_boxes = pd.read_csv(bbox_path, delim_whitespace=True, header=None).astype(int)
        self.df_filenames = pd.read_csv(paths, delim_whitespace=True, header=None)
        filenames = self.df_filenames[1].tolist()

        self.filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            bbox = self.df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            self.filename_bbox[key] = bbox


    def __len__(self):
        return self.dset_size

    def __getitem__(self, item):
        miss_index = (item + random.randint(1, self.dset_size)) % self.dset_size
        item_path = self.paths[item % self.dset_size]
        miss_path = self.paths[miss_index % self.dset_size]
        paths = get_index(item_path, self.images_root, self.text_root, miss=miss_path)

        text_path = paths['text']
        miss_path = paths['miss']
        image_path = paths['image']

        texts = []
        with open(text_path) as f:
            text = f.readlines()
            text = [line.strip() for line in text]
        text = random.sample(text, 1)

        misses = []
        with open(miss_path) as f:
            miss = f.readlines()
            miss = [line.strip() for line in text]
        text = random.sample(miss, 1)

        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        bbox = self.filename_bbox[item_path]

        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        image = image.crop([x1, y1, x2, y2])

        scales = {}
        for scale in self.scales:
            scale_size = scale
            scales[str(scale)] = F.resize(image, (scale_size, scale_size))
            scales[str(scale)] = F.to_tensor(scales[str(scale)])
            scales[str(scale)] = F.normalize(scales[str(scale)], MEAN, VAR, inplace=True)

        return {'text': text, 'miss': miss, 'images': scales}
