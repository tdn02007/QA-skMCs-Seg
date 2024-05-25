import os
from PIL import Image
import numpy as np

data_folder = "./results/cls1_merge/"
data2_folder = "./results/watershed/"
data3_folder = "./results/cls3_merge/"
data4_folder = "./results/skeleton/"

save_folder = "./results/concat/"
os.makedirs(save_folder)

data_list = sorted(os.listdir(data_folder))
data_list2 = sorted(os.listdir(data2_folder))
data_list3 = sorted(os.listdir(data3_folder))
data_list4 = sorted(os.listdir(data4_folder))

for i, data in enumerate(data_list):
    imgs = np.array(Image.open(data_folder + data).convert("L"))
    imgs2 = np.array(Image.open(data2_folder + data_list2[i]).convert("L"))
    imgs3 = np.array(Image.open(data3_folder + data_list3[i]).convert("L"))
    imgs4 = np.array(Image.open(data4_folder + data_list4[i]).convert("L"))

    ### four-image concat
    img_myo = np.where(imgs3[:, :, np.newaxis] > 120, [0, 255, 0], [0, 0, 0])
    img_myo = np.where(imgs[:, :, np.newaxis] > 120, [0, 0, 0], img_myo)
    img_myo = np.where(imgs2[:, :, np.newaxis] > 120, [0, 0, 0], img_myo)
    img_myo = np.where(imgs4[:, :, np.newaxis] > 120, [0, 0, 0], img_myo)

    img_nucl = np.where(imgs[:, :, np.newaxis] > 120, [0, 0, 255], [0, 0, 0])
    img_nucl = np.where(imgs2[:, :, np.newaxis] > 120, [0, 0, 0], img_nucl)
    img_nucl = np.where(imgs4[:, :, np.newaxis] > 120, [0, 0, 0], img_nucl)

    img_wat = np.where(imgs2[:, :, np.newaxis] > 120, [255, 255, 255], [0, 0, 0])
    img_wat = np.where(imgs4[:, :, np.newaxis] > 120, [0, 0, 0], img_wat)

    img_ske = np.where(imgs4[:, :, np.newaxis] > 120, [255, 0, 0], [0, 0, 0])

    final_img = img_myo + img_nucl + img_wat + img_ske

    final_img = Image.fromarray(final_img.astype(np.uint8))

    final_img.save(save_folder + data)

