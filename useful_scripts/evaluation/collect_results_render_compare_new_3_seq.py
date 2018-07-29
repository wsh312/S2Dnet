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
from shutil import copyfile
import shutil
from distutils.dir_util import copy_tree





output_root = "/home/dragon/00rnn_test_output/3-8_all_real_results_select/00all_compare_real_seq/"
input_base = "/home/dragon/00rnn_test_output/3-8_all_real_results_select/"


ply_name = "smvs-clean.ply"
# ply_name = "smvs-S.ply"

#
# selected_names = [1004, 1006, 1011, 1017, 1018, 1025, 1029, 1041, 1121, 1277,
#                   1014, 1096, 1122, 1164, 1239, 1297, 1309, 1354, 1377, 1410, 1441, 1469,
#                   1041, 1099, 1137, 1187, 1250, 1306, 1322, 1356, 1392, 1422, 1444, 1470,
#                   1055, 1101, 1154, 1207, 1260, 1308, 1323, 1358, 1407, 1427, 1445, 1480, 1477, 1473]

# test_names = ["job_submit_101C_re1", "job_submit_101C_re1_pixel", "job_submit_101C_re1_pixel_2", "job_submit_101C_re1_pixel_3",
#               "job_submit_101C_re1_pixel_4", "job_submit_101C_re1_pixel_6", "job_submit_101_re1", "job_submit_102_re2",
#               "job_submit_103_re1", "job_submit_104_re1", "job_submit_105_re1_vgg"]

test_names = ["job_submit_101C_re1_pixel_3_tune", "glossy", "cycle_angle_univer_512_unet", "cycle_angle_univer_512_unet_2",
              "job_submit_101C_re1_pixel_2_tune", "job_submit_101C_re1_pixel", "job_submit_101C_re1_pixel_2", "job_submit_101C_re1_pixel_3",
              "job_submit_101C_re1_pixel_4", "job_submit_101C_re1_pixel_5", "job_submit_101C_re1_pixel_6", "univer_pix2pix_resnet_512",
              "job_submit_101C_re1_pixel_7"]

# selected_names = [1006, 1017, 1041, 1164, 1187, 1306, 1514, 5007, 1543, 1554, 1566, 1623, 1630, 1660, 1698, 1703, 1707,
#                   1711, 1720, 1724, 1726, 1728, 1737, 1758, 1813, 1819, 1879, 1913, 1941, 1961, 2221, 2240, 2291, 2316,
#                   3003, 3118, 3141, 3261, 3267, 3307, 3322, 3361, 3409, 3438, 3446, 3449, 3455, 3472, 3479, 3508, 3529,
#                   3533, 3584, 3635, 3646, 3686, 3688, 3708, 3720, 3778, 3795, 3814, 3869, 3881, 3926, 5009, 5015, 5027,
#                   5044, 5061, 5065, 5072, 5078]

# for test_id in selected_names:

for test_id in range(13):

    for name in test_names:
        input_ply_multi = input_base + name + "/" + str(test_id) + "_recons/" + ply_name


        # input_img_multi = input_base + name  + "/" + str(test_id) + "/" + "output_img-" + str(test_id) + "-cam-" + str(5) + "-pair-1.jpg"

        input_img_multi = input_base + name + "/" + str(test_id) + "/"

        # if name == "glossy":
        #     input_img_multi = input_base + name + "/" + str(test_id) + "/" + "img-" + str(test_id) + "-cam-" + str(5) + "-pair-1.jpg"
        #
        # if name == "diffuse":
        #     input_img_multi = input_base + name + "/" + str(test_id) + "/" + "img-" + str(test_id) + "-cam-" + str(5) + "-pair-0.jpg"

        output_path_base = output_root + str(test_id) + "/"

        output_multi_ply = output_path_base + name + ".ply"

        output_multi_img = output_path_base + name + "/"


        if not os.path.exists(output_path_base):
            os.makedirs(output_path_base)

        if not os.path.exists(output_multi_img):
            os.makedirs(output_multi_img)

        if os.path.exists(input_img_multi):
            # copyfile(input_img_multi, output_multi_img)
            copy_tree(input_img_multi, output_multi_img)
            # shutil.copytree()
        else:
            print "missing: ", input_img_multi

        # if not os.path.exists(input_ply_multi):
        #     continue

        # print os.path.getsize(input_ply_multi)
        # if os.path.getsize(input_ply_multi) < 500000:
        #     print  os.path.getsize(input_ply_multi)
        #     continue

        if os.path.exists(input_ply_multi):
            copyfile(input_ply_multi, output_multi_ply)

        print  output_multi_ply


