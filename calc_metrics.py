import os
import numpy as np
from PIL import Image
from skimage.filters import gaussian

def calculate_iou(image1, image2):
    intersection = np.logical_and(image1, image2).sum()
    union = np.logical_or(image1, image2).sum()
    iou = intersection / union
    return iou

def calculate_dice(image1, image2):
    intersection = np.logical_and(image1, image2).sum()
    dice = (2 * intersection + 1) / (image1.sum() + image2.sum() + 1)
    return dice


class_form = ["cls1", "cls2", "cls3"]

for form in class_form:
    data_folder = f"./results/{form}/"
    label_folder = f"./test_data/{form}/"

    data_list = sorted(os.listdir(data_folder))
    label_list = sorted(os.listdir(label_folder))

    iou_value = 0
    dice_value = 0

    for i, data in enumerate(data_list):
        image1 = np.array(Image.open(data_folder + data).convert("L"))
        image1 = gaussian(image1, sigma=1) > 0.5

        image2 = np.array(Image.open(label_folder + label_list[i]).convert("L"))
        image2 = gaussian(image2, sigma=1) > 0.5

        iou = calculate_iou(image1, image2)
        dice = calculate_dice(image1, image2)

        iou_value += iou
        dice_value += dice

    # print(data)
    print(f"IoU-{form}: {iou_value / len(data_list)}")
    print(f"Dice Coefficient-{form}: {dice_value / len(data_list)}")