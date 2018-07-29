import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, get_transform_patch
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
from PIL import ImageFile
from torchvision.utils import save_image
from torch.autograd import Variable
import torch
import numpy as np
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from skimage import img_as_float
from skimage.measure import compare_mse as mse
from PIL import Image, ImageChops
opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip



# opt.name = "cycle_angle_univer_512_unet"
# opt.which_epoch = "20"
# opt.which_model_netG = "unet_512"
# opt.name = "cycle_angle_real5_2K_single"

opt.no_dropout = True
opt.name = "cycle_angle_univer_512_unet"
opt.which_epoch = "120"
opt.which_model_netG = "unet_512"

# opt.name = "univer_pix2pix_resnet_512"
# opt.which_epoch = "latest"
# opt.which_model_netG = "resnet_9blocks"
# opt.model = "test"
# opt.dataset_mode = 'single'
# opt.norm = "batch"



opt.checkpoints_dir = "/home/dragon/00download_cluster/download_checkpoints/"

model = create_model(opt)

# input_root = "../selected_rnn_test_data/"
# output_root = "../../../00rnn_test_output/select_output/"

# input_root = input_root = "/home/dragon/Dropbox/commute_with_Marco_linux/rnn_test_angel_data/glossy7/"
input_root = "/home/dragon/Dropbox/commute_with_Marco_linux/rnn_test_angel_data/glossy13/"

output_root = "../../../00rnn_test_output/3-18_real_output/"

transform = get_transform(opt)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def mse(imageA, imageB):
    "Calculate the root-mean-square difference between two images"
    diff = ImageChops.difference(imageA, imageB)
    h = diff.histogram()
    sq = (value*((idx%256)**2) for idx, value in enumerate(h))
    sum_of_squares = sum(sq)
    rms = math.sqrt(sum_of_squares/float(imageA.size[0] * imageB.size[1]))
    return rms

sum_sum_err = 0


# selected_names = [1004,  1006,  1011,  1017,  1018,  1025,  1029,  1041,  1121, 1277,
#                   1014, 1096, 1122, 1164, 1239, 1297, 1309, 1354, 1377, 1410, 1441, 1469,
#                   1041, 1099, 1137, 1187, 1250, 1306, 1322, 1356, 1392, 1422, 1444, 1470,
#                   1055, 1101, 1154, 1207, 1260, 1308, 1323, 1358, 1407, 1427, 1445, 1484, 1480, 1477, 1473         ]

for test_id in range(9):

    sum_err = 0

    output_path_base = output_root + opt.name + "/" + str(test_id) + "/"

    input_path_base = input_root + str(test_id) + "/"

    list_dirs = os.walk(input_path_base)

    input_paths = []

    for root, dirs, files in list_dirs:
        for f in sorted(files):
            input_paths.append(input_path_base + f)

    print  test_id

    for cam_id in range(0, len(input_paths)):

        # input_path = input_root + str(test_id) + "/" + "img-" + str(test_id) + "-cam-" + str(cam_id) + "-pair-1.png"

        # input_path = input_root + str(test_id) + "/" + "angel34_512_" + str(cam_id) + ".JPG"

        input_path = input_paths[cam_id]


        if not os.path.exists(input_path):
            continue

        if not os.path.exists(output_path_base):
            os.makedirs(output_path_base)

        output_path = output_path_base + "output_img-" + str(test_id) + "-cam-" + str(cam_id) + "-pair-1.png"

        # if os.path.exists(output_path):
        #     continue

        A_img = Image.open(input_path).convert('RGB')
        A = transform(A_img)
        # A = torch.Tensor(A)
        A = A.unsqueeze(0)
        A = Variable(A, volatile=True).cuda()

        # fixed_x = []
        # fixed_x.append(A)
        # fixed_x = torch.cat(fixed_x, dim=0)
        #
        # fixed_x = Variable(fixed_x, volatile=True).cuda()


        B = model.netG_A(A)
        # B = model.netG(A)


        save_image(denorm(B.data), output_path, nrow=1, padding=0)


        # print  output_path
