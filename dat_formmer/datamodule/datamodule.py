import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from zipfile import ZipFile

import numpy as np
import lightning as L
import torch
from dat_formmer.datamodule.dataset import CROHMEDataset
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader

from .vocab import vocab

Data = List[Tuple[str, Image.Image, List[str]]]

MAX_SIZE = 32e4  # change here accroading to your GPU memory

# load data
def data_iterator(
    data: Data,
    batch_size: int,
    batch_Imagesize: int = MAX_SIZE,
    maxlen: int = 200,
    maxImagesize: int = MAX_SIZE,
):
    fname_batch = []
    feature_batch = []
    label_batch = []
    feature_total = []
    label_total = []
    fname_total = []
    biggest_image_size = 0

    data.sort(key=lambda x: x[1].size[0] * x[1].size[1])

    i = 0
    # max_width, max_height, max_length = 0, 0, 0
    for fname, fea, lab in data:
        # size = fea.size[0] * fea.size[1]
        h, w = fea.size[0], fea.size[1]
        size = h * w
        fea = np.array(fea)
        if size > biggest_image_size:
            biggest_image_size = size
        batch_image_size = biggest_image_size * (i + 1)
        if len(lab) > maxlen:
            print("sentence", i, "length bigger than", maxlen, "ignore")
        elif size > maxImagesize:
            print(
                f"image: {fname} size: {fea.shape[0]} x {fea.shape[1]} =  bigger than {maxImagesize}, ignore"
            )
        # elif w * max_width > 1600 * 320:
        #     print(f"image: {fname} width {w} too large, ignore")
        #     # continue
        # elif h * max_height > 1600 * 320:
        #     print(f"image: {fname} height {h} too large, ignore")
        #     # continue
        else:
            if batch_image_size > batch_Imagesize or i == batch_size:  # a batch is full
                fname_total.append(fname_batch)
                feature_total.append(feature_batch)
                label_total.append(label_batch)
                i = 0
                biggest_image_size = size
                fname_batch = []
                feature_batch = []
                label_batch = []
                fname_batch.append(fname)
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1
            else:
                fname_batch.append(fname)
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1
            # max_height = h if h > max_height else max_height
            # max_width = w if w > max_width else max_width

    # last batch
    fname_total.append(fname_batch)
    feature_total.append(feature_batch)
    label_total.append(label_batch)
    print("total ", len(feature_total), "batch data loaded")
    return list(zip(fname_total, feature_total, label_total))


def extract_data(archive: ZipFile, dir_name: str) -> Data:
    """Extract all data need for a dataset from zip archive

    Args:
        archive (ZipFile):
        dir_name (str): dir name in archive zip (eg: train, test_2014......)

    Returns:
        Data: list of tuple of image and formula
    """
    with archive.open(f"data/{dir_name}/caption.txt", "r") as f:
        captions = f.readlines()
    # if dir_name == "train":
    #     # remove the first line, which is the header
    #     captions = captions[0:100]
    # else:
    #     # remove the first line, which is the header
    #     captions = captions[0:50]
    data = []
    for line in captions:
        tmp = line.decode().strip().split()
        img_name = tmp[0]
        formula = tmp[1:]
        with archive.open(f"data/{dir_name}/img/{img_name}.bmp", "r") as f:
            # move image to memory immediately, avoid lazy loading, which will lead to None pointer error in loading
            img = Image.open(f).copy()
        data.append((img_name, img, formula))

    print(f"Extract data from: {dir_name}, with data size: {len(data)}")

    return data


@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    indices: List[List[int]]  # [b, l]
    structure: List[List[int]] # [b, l]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            indices=self.indices,
            structure=self.structure,
        )


def collate_fn(batch):
    assert len(batch) == 1
    batch = batch[0]
    fnames = batch[0]
    images_x = batch[1]
    seqs_y = [vocab.words2indices(x) for x in batch[2]]
    structure_y = [build_cursor_pos(x) for x in batch[2]]

    heights_x = [s.size(1) for s in images_x]
    widths_x = [s.size(2) for s in images_x]

    n_samples = len(heights_x)
    max_height_x = max(heights_x)
    max_width_x = max(widths_x)

    x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
    x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)
    for idx, s_x in enumerate(images_x):
        x[idx, :, : heights_x[idx], : widths_x[idx]] = s_x
        x_mask[idx, : heights_x[idx], : widths_x[idx]] = 0

    # return fnames, x, x_mask, seqs_y
    return Batch(fnames, x, x_mask, seqs_y, structure_y)


def build_dataset(archive, folder: str, batch_size: int):
    data = extract_data(archive, folder)
    return data_iterator(data, batch_size)


class CROHMEDatamodule(L.LightningDataModule):
    def __init__(
        self,
        zipfile_path: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../data.zip",
        test_year: str = "2014",
        train_batch_size: int = 8,
        eval_batch_size: int = 4,
        num_workers: int = 5,
        scale_aug: bool = False,
    ) -> None:
        super().__init__()
        assert isinstance(test_year, str)
        self.zipfile_path = zipfile_path
        self.test_year = test_year
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.scale_aug = scale_aug

        print(f"Load data from: {self.zipfile_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        with ZipFile(self.zipfile_path) as archive:
            if stage == "fit" or stage is None:
                self.train_dataset = CROHMEDataset(
                    build_dataset(archive, "train", self.train_batch_size),
                    True,
                    self.scale_aug,
                )
                self.val_dataset = CROHMEDataset(
                    build_dataset(archive, self.test_year, self.eval_batch_size),
                    False,
                    self.scale_aug,
                )
            if stage == "test" or stage is None:
                self.test_dataset = CROHMEDataset(
                    build_dataset(archive, self.test_year, self.eval_batch_size),
                    False,
                    self.scale_aug,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True, # 配合GPU使用时，务必开启
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True, # 配合GPU使用时，务必开启
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True, # 配合GPU使用时，务必开启
        )


CURSOR_VOCAB = {
            '<pad>': 0,
            '<sos>': 1,  # 开始符
            '<eos>': 2,  # 结束符
            # 位置标记
            'default': 3,     # 默认位置（无嵌套结构）
            'frac_num': 4,    # 分子
            'frac_den': 5,    # 分母
            'sqrt_inner': 6,  # 根号内
            'subscript': 7,   # 下标
            'superscript': 8, # 上标
            'integral': 9,    # 积分体
            'lim': 10          # 极限参数
        }


def build_cursor_pos(tokens, cursor_vocab=CURSOR_VOCAB):
    cursor_stack = ['default']
    cursor_positions = []

    # if cursor_vocab is None:
        

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok == '\\frac':
            cursor_stack.append('frac_num')
            cursor_positions.append(cursor_vocab['default'])  # \frac 本身
            i += 1
        elif tok == '{':
            current_context = cursor_stack[-1]
            cursor_positions.append(cursor_vocab[current_context])  # “{”继承当前结构
            i += 1
        elif tok == '}':
            cursor_positions.append(cursor_vocab[cursor_stack[-1]])
            # 判断结构是否要弹栈（假设一个结构体包含一对 {}）
            if cursor_stack[-1] in ['frac_num']:
                cursor_stack[-1] = 'frac_den'  # 进入分母
            elif cursor_stack[-1] in ['frac_den', 'sqrt_inner', 'subscript', 'superscript', 'integral', 'lim']:
                cursor_stack.pop()
            i += 1
        elif tok == '\\sqrt':
            cursor_stack.append('sqrt_inner')
            cursor_positions.append(cursor_vocab['default'])
            i += 1
        elif tok == '_':
            cursor_stack.append('subscript')
            cursor_positions.append(cursor_vocab['default'])
            i += 1
        elif tok == '^':
            cursor_stack.append('superscript')
            cursor_positions.append(cursor_vocab['default'])
            i += 1
        elif tok in ['\\int', '\\sum']:
            cursor_stack.append('integral')
            cursor_positions.append(cursor_vocab['default'])
            i += 1
        elif tok == '\\lim':
            cursor_stack.append('lim')
            cursor_positions.append(cursor_vocab['default'])
            i += 1
        else:
            cursor_positions.append(cursor_vocab[cursor_stack[-1]])
            i += 1
    # cursor_positions = [0] + cursor_positions  # 添加起始位置的默认值

    return cursor_positions
