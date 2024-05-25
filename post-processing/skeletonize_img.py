import os
from PIL import Image
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, dilation, remove_small_holes, remove_small_objects, square
from skimage.filters import gaussian
import numpy as np
import cv2

myotube_folder = "./results/cls3_merge/"
skeleton_folder = "./results/skeleton/"

os.makedirs(skeleton_folder, exist_ok=True)

data_list = os.listdir(myotube_folder)

for data in data_list:
    imgs = np.array(Image.open(myotube_folder + data).convert("L"))

    denoise_img = (gaussian(remove_small_holes(imgs), sigma=1) > 0.5).view(np.uint8) * 255

    skeleton_img = skeletonize(denoise_img, method="lee").view(np.uint8)

    final_img = dilation(skeleton_img, square(7)) > 0.5

    final_img = remove_small_objects(final_img).view(np.uint8) * 255

    cv2.imwrite(skeleton_folder + data, final_img)


    