import os
import subprocess
import sys

import math
import numpy as np
import random
import time
from math import fabs
from shutil import copyfile
from PIL import Image
from PIL import ImageOps


rootDir = os.path.dirname(os.path.abspath(__file__))


first_img_id = 0
last_img_id = 200



img_size = 512
for img_id in range(first_img_id, last_img_id):

    img0_path = rootDir + "/output/" + "img-" + str(img_id) + "-pair-" + str(0) + ".png"
    img1_path = rootDir + "/output/" + "img-" + str(img_id) + "-pair-" + str(1) + ".png"
    img_concat_path = rootDir + "/output_concat/" + "img-" + str(img_id) + "_concat.png"
    img_concat_flip_path = rootDir + "/output_concat/" + "img-" + str(img_id) + "_concat_flip.png"

    if (os.path.exists(img_concat_path)):
        continue
    if (os.path.exists(img0_path) is False):
        continue
    if (os.path.exists(img1_path) is False):
        continue

    print(str(img_id))
    img0 = Image.open(img0_path)
    img1 = Image.open(img1_path)

    img0_flip = ImageOps.mirror(img0)
    img1_flip = ImageOps.mirror(img1)


    img_concat = Image.new('RGB', (img_size*2, img_size))
    img_concat.paste(img1, (0, 0))
    img_concat.paste(img0, (img_size, 0))
    img_concat.save(img_concat_path)

    # img_concat_flip = Image.new('RGB', (img_size*2, img_size))
    # img_concat_flip.paste(img1_flip, (0, 0))
    # img_concat_flip.paste(img0_flip, (img_size, 0))
    # img_concat_flip.save(img_concat_flip_path)