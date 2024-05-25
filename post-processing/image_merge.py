import os
from PIL import Image
import math

data_form = ["cls1", "cls2", "cls3"]

for form in data_form:
    data_dir = f"./results/{form}"
    data_list = sorted(os.listdir(data_dir))

    os.makedirs(f"./results/{form}_merge/", exist_ok=True)

    for index in range(int(len(data_list)/16)):
        new_image = Image.new('L', (2048, 2048))
        for num in range(16):
            img = Image.open(data_dir + "/" + data_list[index * 16 + num]).convert("L")
            row = math.floor(num / 4) * 512
            col = (num % 4) * 512
            new_image.paste(img, (row, col))

        new_image.save(f"./results/{form}_merge/" +
                    str(index) + ".tif", "TIFF")
