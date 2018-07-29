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

# input_root = "D:\Dropbox\Dropbox\Matan_Shihao_commute/6-13_new_render_multiview_images\output_color_multi"
input_root = "./rnn_cycle_output/"
output_root = "./rnn_cycle_output_transfer/"

first_img_id = 99
last_img_id = 109
img_size = 512

img0_path = input_root + "test_img-" + str(first_img_id) + "_concat_fake_B.png"
img0 = Image.open(img0_path)

img_concat_path1 = output_root + "img-cam-" + str(first_img_id) + "_rnn.jpg"
img_concat_path2 = output_root + "img-cam-" + str(first_img_id+1) + "_rnn.jpg"
img_concat_path3 = output_root + "img-cam-" + str(first_img_id+2) + "_rnn.jpg"

img_concat1 = Image.new('RGB', (img_size, img_size))
img_concat2 = Image.new('RGB', (img_size, img_size))
img_concat3 = Image.new('RGB', (img_size, img_size))

area = (img_size * 0, 0, img_size * 1, img_size)
img_concat1.paste(img0.crop(area), (0, 0))

area = (img_size * 1, 0, img_size * 2, img_size)
img_concat2.paste(img0.crop(area), (0, 0))

area = (img_size * 2, 0, img_size * 3, img_size)
img_concat3.paste(img0.crop(area), (0, 0))

img_concat1.save(img_concat_path1)
img_concat2.save(img_concat_path2)
img_concat3.save(img_concat_path3)



for img_id in range(first_img_id+1, last_img_id):

    img0_path = input_root + "test_img-" + str(img_id) + "_concat_fake_B.png"
    img_concat_path = output_root + "img-cam-" + str(img_id+2) + "_rnn.jpg"

    #if (os.path.exists(img_concat_path)):
    #    continue
    if (os.path.exists(img0_path) is False):
        continue

    print(str(img_id))
    img0 = Image.open(img0_path)

    img_concat = Image.new('RGB', (img_size, img_size))
    area = (img_size * 2, 0, img_size * 3, img_size)
    img_concat.paste(img0.crop(area), (0, 0))

    img_concat.save(img_concat_path)



