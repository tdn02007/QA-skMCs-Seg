import os
from PIL import Image
import numpy as np
from skimage.exposure import rescale_intensity


data_folder = "./raw_data/train/inp2/"

data_list = os.listdir(data_folder)

save_folder = f"./data/train_data/inp2_data/"
os.makedirs(save_folder, exist_ok=True)

for data in data_list:
    imgs = Image.open(data_folder + data).convert("L").resize((2048, 2048))
    imgs = np.array(imgs)

    if "17M" in data:
        imgs = rescale_intensity(imgs, in_range=(0, 25), out_range=(0, 255))
    else:
        imgs = rescale_intensity(imgs)

    data_name = data.split('.')[0]

    for j in range(4):
        for i in range(4):
            contrast_image_crop = imgs[i*512:i*512 + 512, j*512:j*512+512]
            contrast_image_crop = Image.fromarray(contrast_image_crop.astype(np.uint8))

            contrast_image_crop.save(save_folder + data_name + str(j) + "-" + str(i) + ".tif")