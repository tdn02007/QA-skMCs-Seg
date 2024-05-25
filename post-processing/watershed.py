import cv2
import numpy as np
from PIL import Image
from skimage.morphology import dilation, square
import os
import numpy as np

boundary_folder = "../results/cls1_merge/"
marker_folder = "../results/cls2_merge/"
save_folder = "../results/watershed/"

os.makedirs(save_folder, exist_ok=True)

boundary_dir = sorted(os.listdir(boundary_folder))
marker_dir = sorted(os.listdir(marker_folder))

for i, f in enumerate(boundary_dir):
    img2 = cv2.imread(boundary_folder + f)
    img3 = cv2.imread(marker_folder + marker_dir[i])

    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(gray2, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    sure_fg = gray3

    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    
    markers = markers + 1

    markers[unknown == 255] = 0

    markers = cv2.watershed(img2, markers)

    img2[markers == -1] = [255, 0, 0]

    data = np.zeros_like(markers)
    
    data[markers == -1] = 255
    
    data = dilation(data, square(3))

    final_img = Image.fromarray(data.astype(np.uint8))

    final_img.save(save_folder + f)


