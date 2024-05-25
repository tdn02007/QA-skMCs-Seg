import os

from glob import glob
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset


class DatasetLoader(Dataset):
    def __init__(self):
        self.input_dir1 = "./data/train_data_sample/inp1_data/"
        self.input_dir2 = "./data/train_data_sample/inp2_data/"
        self.label_dir1 = "./data/train_data_sample/cls1_data/"
        self.label_dir2 = "./data/train_data_sample/cls2_data/"
        self.label_dir3 = "./data/train_data_sample/cls3_data/"

        self.input_images1 = sorted(os.listdir(self.input_dir1))
        self.input_images2 = sorted(os.listdir(self.input_dir2))
        self.label_images1 = sorted(os.listdir(self.label_dir1))
        self.label_images2 = sorted(os.listdir(self.label_dir2))
        self.label_images3 = sorted(os.listdir(self.label_dir3))

    @classmethod
    def preprocess(cls, pil_img, train):
        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        if train:
            img_trans = img_trans / 255.

        return img_trans

    def __len__(self):
        return len(self.input_images1)

    def __getitem__(self, index):
        image_name1 = self.input_images1[index]
        image_name2 = self.input_images2[index]

        input_file1 = glob(self.input_dir1 + image_name1)
        input_file2 = glob(self.input_dir2 + image_name2)

        input_image1 = np.array(Image.open(input_file1[0]))
        input_image2 = np.array(Image.open(input_file2[0]))

        input_image1 = np.reshape(
            input_image1, (input_image1.shape[0], input_image1.shape[1], 1))
        input_image2 = np.reshape(
            input_image2, (input_image2.shape[0], input_image2.shape[1], 1))

        input_image = np.concatenate((input_image1, input_image2), axis=2)

        input_image = self.preprocess(input_image, True)

        label_name1 = self.label_images1[index]
        label_name2 = self.label_images2[index]
        label_name3 = self.label_images3[index]

        label_file1 = glob(self.label_dir1 + label_name1)
        label_file2 = glob(self.label_dir2 + label_name2)
        label_file3 = glob(self.label_dir3 + label_name3)

        label_image1 = np.array(Image.open(label_file1[0]))
        label_image2 = np.array(Image.open(label_file2[0]))
        label_image3 = np.array(Image.open(label_file3[0]))

        label_image1 = np.reshape(
            label_image1, (label_image1.shape[0], label_image1.shape[1], 1))
        label_image2 = np.reshape(
            label_image2, (label_image2.shape[0], label_image2.shape[1], 1))
        label_image3 = np.reshape(
            label_image3, (label_image3.shape[0], label_image3.shape[1], 1))

        label_image1 = np.where(label_image1 == 255, 1, 0)
        label_image2 = np.where(label_image2 == 255, 1, 0)
        label_image3 = np.where(label_image3 == 255, 1, 0)

        label_image = np.concatenate((label_image1, label_image2), axis=2)
        label_image = np.concatenate((label_image, label_image3), axis=2)

        label_image = self.preprocess(label_image, False)

        return {
            "input_image": torch.from_numpy(input_image).type(torch.FloatTensor),
            "label_image": torch.from_numpy(label_image).type(torch.FloatTensor),
        }
