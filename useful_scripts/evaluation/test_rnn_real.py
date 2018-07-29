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
opt.no_dropout = True

opt.checkpoints_dir = "/home/dragon/00download_cluster/download_checkpoints/"
# opt.checkpoints_dir = "/home/dragon/00download_cluster/download_checkpoints/3-4_compare_checkpoints/"

# opt.checkpoints_dir = "/media/dragon/New Volume/3-5_save_data/3-4-backup-checkpoints/"

# opt.checkpoints_dir = "/media/dragon/New Volume/3-5_save_data/3-7_save_checkpoints/"


# selected_epoch = ["95", "105", "115", "125", "135"]

# selected_epoch = ["55", "65", "75", "latest"]

selected_epoch = ["latest"]




input_root = "/home/dragon/Dropbox/commute_with_Marco_linux/rnn_test_angel_data/glossy14/"
output_root = "../../../00rnn_test_output/3-19_real_output/"

transform = get_transform(opt)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

img_size = 512


def mse(imageA, imageB):
    "Calculate the root-mean-square difference between two images"
    diff = ImageChops.difference(imageA, imageB)
    h = diff.histogram()
    sq = (value*((idx%256)**2) for idx, value in enumerate(h))
    sum_of_squares = sum(sq)
    rms = math.sqrt(sum_of_squares/float(imageA.size[0] * imageB.size[1]))
    return rms




test_names = []
# test_names.append("job_submit_53_multiD_unet1024_patch128old")

# test_names.append("xrnn_512_256_multiD_vgg")
# test_names.append("t15_unet1024_patch128_multiD_vgg_sgd")


# test_names.append("t31_unet1024_patch128_multiD_vgg_sgd")
# test_names.append("t35_unet512_patch128_multiD_vgg_sgd")
# test_names.append("t37_unet1024_patch256new_multiD_vgg_sgd")


# test_names.append("job_submit_32_multiD_unet512_patch256_vgg15")
# test_names.append("job_submit_43_multiD_unet1024_patch256_vgg_lambda10")
# test_names.append("job_submit_44_multiD_unet1024_patch256_vgg_lambda10")
# test_names.append("job_submit_52_multiD_unet512_patch128old")
# test_names.append("job_submit_61_multiD_unet512_patch128old")

# test_names.append("job_submit_101C_re1")
# test_names.append("job_submit_22_multiD_unet512_patch128_vgg_lambda10")


# test_names.append("t43_reconv1")
#
# test_names.append("t45_reconv1_pixel")
# test_names.append("t40_unet1024_vgg0")
# test_names.append("t44_reconv2")

# test_names.append("t39_unet1024_gd10")

# test_names.append("")


# test_names.append("job_submit_101C_re1_pixel_6_tune")
# test_names.append("job_submit_101C_re1_pixel_7")
# test_names.append("job_submit_105_re1_vgg")
# test_names.append("job_submit_105_re1_vgg_tune")

#

# test_names.append("job_submit_101C_re1_pixel")
# # test_names.append("job_submit_101C_re1_pixel_2")
# # test_names.append("job_submit_101C_re1_pixel_6")
#
# # test_names.append("job_submit_101C_re1_pixel_6_tune")
# # test_names.append("job_submit_101C_re1_pixel_3_tune")
#
# #
#
# #
# # test_names.append("job_submit_101C_re1_pixel_6_tune")
# # test_names.append("job_submit_101C_re1_pixel_2_tune")
# # test_names.append("job_submit_101C_re1_pixel_3_tune")
#
# test_names.append("job_submit_101C_re1_pixel")
# # test_names.append("job_submit_101C_re1_pixel_2")
# # test_names.append("job_submit_101C_re1_pixel_3")
# test_names.append("job_submit_101C_re1_pixel_4")
# # test_names.append("job_submit_101C_re1_pixel_5")
# test_names.append("job_submit_101C_re1_pixel_6")
# # test_names.append("job_submit_101C_re1_pixel_7")


# test_names.append("t43_reconv1")
# test_names.append("t44_reconv2")
# test_names.append("t40_unet1024_vgg0")
# test_names.append("t41_unet1024_gd0")
# test_names.append("t45_reconv1_pixel")


test_names.append("job_submit_101C_re1_pixel")
test_names.append("job_submit_101C_re1_pixel_2_tune")
test_names.append("job_submit_101C_re1_pixel_2")
test_names.append("job_submit_101C_re1_pixel_3")
test_names.append("job_submit_101C_re1_pixel_4")
test_names.append("job_submit_101C_re1_pixel_5")
test_names.append("job_submit_101C_re1_pixel_6")
test_names.append("job_submit_101C_re1_pixel_7")
# #
test_names.append("job_submit_101C_re1_pixel_6_tune")
test_names.append("job_submit_101C_re1_pixel_3_tune")


for epoch in selected_epoch:

    opt.which_epoch = epoch

    for test_name in test_names:

        sum_sum_err = 0

        opt.name = test_name
        opt.which_model_netG = "unet_512"

        if "unet1024" in test_name:
            opt.which_model_netG = "unet_1024"

        if "xrnn_512" in test_name:
            opt.which_model_netG = "xrnn_512"

        if test_name == "job_submit_61_multiD_unet1024_patch128old":
            opt.which_model_netG = "unet_512"

        if "_re1" in test_name or "_reconv1" in test_name:
            opt.which_model_netG = "unet_512_Re1"

        if "_re2" in test_name or"_reconv2" in test_name:
            opt.which_model_netG = "unet_512_Re2"

        if "_pixel" in test_name:
            opt.norm = "pixel"
        else:
            opt.norm = "instance"


        model = create_model(opt)


        # selected_names = [1101, 1164]

        for test_id in range(0, 1):

            sum_err = 0
            count = 0

            output_path_base = output_root + opt.name + "/" + str(test_id) + "/"
            # output_path_base = output_root + str(test_id) + "/"

            if not os.path.exists(output_path_base):
                os.makedirs(output_path_base)


            input_path_base = input_root + str(test_id) + "/"

            list_dirs = os.walk(input_path_base)

            input_paths = []

            for root, dirs, files in list_dirs:
                for f in sorted(files):
                    input_paths.append(input_path_base + f)

            print test_id

            for cam_id in range(0, len(input_paths)-2):
                # count += 1
                # print cam_id
                input_path1 = input_paths[cam_id]
                input_path2 = input_paths[cam_id+1]
                input_path3 = input_paths[cam_id+2]


                if not os.path.exists(input_path1):
                    continue
                if not os.path.exists(input_path2):
                    continue
                if not os.path.exists(input_path3):
                    continue




                # target_path = input_root + str(test_id) + "/" + "img-" + str(test_id) + "-cam-" + str(cam_id) + "-pair-0.png"
                # img_target = Image.open(target_path).convert('RGB')

                output_path = output_path_base + "output_img-" + str(test_id) + "-cam-" + str(cam_id+1) + "-pair-1.png"


                # if os.path.exists(output_path):
                #     continue

                A_img1 = Image.open(input_path1).convert('RGB')
                A_img2 = Image.open(input_path2).convert('RGB')
                A_img3 = Image.open(input_path3).convert('RGB')

                # err_input = mse(A_img2, img_target)
                # print  ["input error", str(err_input)]

                img_concat = Image.new('RGB', (img_size * 3, img_size * 1))

                img_concat.paste(A_img1, (0 * img_size, 0))
                img_concat.paste(A_img2, (1 * img_size, 0))
                img_concat.paste(A_img3, (2 * img_size, 0))

                A = transform(img_concat)
                # A = torch.Tensor(A)
                A = A.unsqueeze(0)
                A = Variable(A, volatile=True).cuda()

                B = model.netG_A(A)

                B_img = denorm(B.data)

                # B_img = B_img[:,:,:,512:512*2]
                #
                #
                # save_image(B_img, output_path, nrow=1, padding=0)
                #
                # B_img = denorm(B.data)

                B_img_save = B_img[:, :, :, 512:512 * 2]

                save_image(B_img_save, output_path, nrow=1, padding=0)

                if cam_id == 0:
                    output_path = output_path_base + "output_img-" + str(test_id) + "-cam-" + str(
                        cam_id) + "-pair-1.png"
                    B_img_save = B_img[:, :, :, 0:512]
                    save_image(B_img_save, output_path, nrow=1, padding=0)

                if cam_id == len(input_paths)-3:
                    output_path = output_path_base + "output_img-" + str(test_id) + "-cam-" + str(
                        cam_id + 2) + "-pair-1.png"
                    B_img_save = B_img[:, :, :, 512 * 2:512 * 3]
                    save_image(B_img_save, output_path, nrow=1, padding=0)

                # img_res = Image.open(output_path).convert('RGB')
                # err = mse(img_res, img_target)
                # sum_err += err


                # print  ["output error", str(err) ]


                # print  output_path

            # avg_err = sum_err / count
            # if avg_err > 0:
            #     print "avg_err: "
            #     print  avg_err
            #
            # output_txt_path = output_path_base + "error_" + str(avg_err) + ".txt"
            # text_file = open(output_txt_path, "w")
            #
            # text_file.write(str(avg_err))
            # text_file.close()

            # sum_sum_err += avg_err
        #

        # avg_err = sum_sum_err / 22
        # if avg_err > 0:
        #     print "final avg_err: "
        #     print  avg_err
        #
        # output_txt_path = output_root + opt.name + "/" + str(opt.which_epoch) + "/" + "error_" + str(avg_err) + ".txt"
        # text_file = open(output_txt_path, "w")
        #
        # text_file.write(str(avg_err))
        # text_file.close()


