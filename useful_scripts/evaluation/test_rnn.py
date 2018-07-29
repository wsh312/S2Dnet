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
# import cv2
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

# opt.checkpoints_dir = "/home/dragon/00download_cluster/download_checkpoints/"
# opt.checkpoints_dir = "/home/dragon/00download_cluster/download_checkpoints/3-4_compare_checkpoints/"
opt.checkpoints_dir = "./checkpoints/"

# opt.checkpoints_dir = "/media/dragon/New Volume/3-5_save_data/3-7_save_checkpoints/"


input_root = "../selected_rnn_test_data/"
# input_root = "/media/dragon/My Passport/output_test_new/"


output_root = "../../../00rnn_test_output/select_output_50/"
# output_root = "../../../00rnn_test_output/select_output_15/"


# output_root = "../../../00rnn_test_output/select_output_10_more/"

# start_id = 1
# end_id = 49

# start_id = 1
# end_id = 9

start_id = 1
end_id = 50

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


# test_names.append("job_submit_22_multiD_unet512_patch128_vgg_lambda10")
# test_names.append("job_submit_23_multiD_unet512_patch256_vgg_lambda10")
# test_names.append("job_submit_61_multiD_unet512_patch128old")
# test_names.append("job_submit_82")
# test_names.append("job_submit_81")

# test_names.append("job_submit_32_multiD_unet512_patch256_vgg15")
# test_names.append("job_submit_43_multiD_unet1024_patch256_vgg_lambda10")
# test_names.append("job_submit_44_multiD_unet1024_patch256_vgg_lambda10")
# test_names.append("job_submit_52_multiD_unet512_patch128old")

# test_names.append("job_submit_62_t15_multiD_unet1024")


# test_names.append("job_submit_101_re1")
# test_names.append("job_submit_101C_re1")
# test_names.append("job_submit_101C_re1_pixel")
# test_names.append("job_submit_103_re1")
# test_names.append("job_submit_105_re1_vgg")
# test_names.append("job_submit_104_re1")
# test_names.append("job_submit_102_re2")
#
# test_names.append("job_submit_101C_re1_pixel_2")
# test_names.append("job_submit_101C_re1_pixel_3")
# test_names.append("job_submit_101C_re1_pixel_4")
# test_names.append("job_submit_101C_re1_pixel_5")
# test_names.append("job_submit_101C_re1_pixel_6")
# # test_names.append("job_submit_101C_re1_finetune")
#
# test_names.append("job_submit_101C_re1_pixel_7")
#
test_names.append("job_submit_101C_re1_pixel")
test_names.append("job_submit_101C_re1_pixel_2")
# test_names.append("job_submit_101C_re1_pixel_3")
test_names.append("job_submit_101C_re1_pixel_4")
test_names.append("job_submit_101C_re1_pixel_5")
# test_names.append("job_submit_101C_re1_pixel_6")
# test_names.append("job_submit_101C_re1_pixel_7")

# test_names.append("job_submit_101C_re1_pixel")
# test_names.append("job_submit_101C_re1_pixel_4")
# test_names.append("job_submit_101C_re1_pixel_6")

total_error = 0.
total_count = 0

for test_name in test_names:

    sum_sum_err = 0

    opt.name = test_name
    opt.which_epoch = "latest"
    opt.which_model_netG = "unet_512"

    # print opt.name

    if "unet1024" in test_name:
        opt.which_model_netG = "unet_1024"

    if "xrnn_512" in test_name:
        opt.which_model_netG = "xrnn_512"

    if test_name == "job_submit_61_multiD_unet1024_patch128old":
        opt.which_model_netG = "unet_512"

    if "_re1" in test_name:
        opt.which_model_netG = "unet_512_Re1"

    if "_re2" in test_name:
        opt.which_model_netG = "unet_512_Re2"

    if "_pixel" in test_name:
        opt.norm = "pixel"
    else:
        opt.norm = "instance"

    model = create_model(opt)

    # selected_names = [1004,  1006,  1011,  1017,  1018,  1025,  1029,  1041,  1121, 1277,
    #                   1014, 1096, 1122, 1164, 1239, 1297, 1309, 1354, 1377, 1410, 1441, 1469,
    #                   1041, 1099, 1137, 1187, 1250, 1306, 1322, 1356, 1392, 1422, 1444, 1470,
    #                   1055, 1101, 1154, 1207, 1260, 1308, 1323, 1358, 1407, 1427, 1445, 1484, 1480, 1477, 1473         ]

    # selected_names = [1004,  1006,  1011,  1017,  1018,  1025,  1029,  1041,  1121, 1277,
    #                   1014, 1096, 1122, 1164, 1239, 1297, 1309, 1354, 1377, 1410, 1441, 1469,
    #                   1041, 1099, 1137, 1187, 1250, 1306, 1322, 1356, 1392, 1422, 1444, 1470,
    #                   1055, 1101, 1154, 1207, 1260, 1308, 1323, 1358, 1407, 1427, 1445, 1480, 1477, 1473]

    # selected_names = [1006,  1017,  1041 , 1096 , 1099 , 1164 , 1187, 1239,  1306 , 1322,  1444]

    selected_names = [1164]

    # selected_names = [1006 , 1017 , 1041 , 1164 , 1187 , 1306 , 1514 , 5007 , 1543 , 1554 , 1566 , 1623 , 1630 , 1660 , 1698 , 1703 , 1707 , 1711 , 1720 , 1724 , 1726 , 1728 , 1737 , 1758 , 1813 , 1819 , 1879 , 1913 , 1941 , 1961 , 2221 , 2240 , 2291 , 2316 , 3003 , 3118 , 3141 , 3261 , 3267 , 3307 , 3322 , 3361 , 3409 , 3438 , 3446 , 3449 , 3455 , 3472 , 3479 , 3508 , 3529 , 3533 , 3584 , 3635 , 3646 , 3686 , 3688 , 3708 , 3720 , 3778 , 3795 , 3814 , 3869 , 3881 , 3926 , 5009 , 5015 , 5027 , 5044 , 5061 , 5065 , 5072 , 5078]

    # selected_names = [1101, 1164]

    for test_id in selected_names:
    # for test_id in range(3000, 4000):

        sum_err = 0
        count = 0

        # output_path_base = output_root + opt.name + "/" + str(opt.which_epoch) + "/" + str(test_id) + "/"
        output_path_base = output_root + opt.name + "/" + str(test_id) + "/"

        # print test_id

        for cam_id in range(start_id, end_id+1):
        # for cam_id in range(1, 51):
            count += 1

            total_count += 1

            # print cam_id
            input_path2 = input_root + str(test_id) + "/" + "img-" + str(test_id) + "-cam-" + str(cam_id) + "-pair-1.png"
            input_path1 = input_root + str(test_id) + "/" + "img-" + str(test_id) + "-cam-" + str(cam_id-1) + "-pair-1.png"
            input_path3 = input_root + str(test_id) + "/" + "img-" + str(test_id) + "-cam-" + str(cam_id+1) + "-pair-1.png"


            if not os.path.exists(input_path1):
                continue
            if not os.path.exists(input_path2):
                continue
            if not os.path.exists(input_path3):
                continue

            if not os.path.exists(output_path_base):
                os.makedirs(output_path_base)


            target_path = input_root + str(test_id) + "/" + "img-" + str(test_id) + "-cam-" + str(cam_id) + "-pair-0.png"
            img_target = Image.open(target_path).convert('RGB')

            output_path = output_path_base + "output_img-" + str(test_id) + "-cam-" + str(cam_id) + "-pair-1.png"


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

            B_img_save = B_img[:,:,:,512:512*2]

            save_image(B_img_save, output_path, nrow=1, padding=0)

            if cam_id == start_id:
                output_path2 = output_path_base + "output_img-" + str(test_id) + "-cam-" + str(cam_id-1) + "-pair-1.png"
                B_img_save = B_img[:, :, :, 0:512]
                save_image(B_img_save, output_path2, nrow=1, padding=0)


            if cam_id == end_id:
                output_path3 = output_path_base + "output_img-" + str(test_id) + "-cam-" + str(cam_id+1) + "-pair-1.png"
                B_img_save = B_img[:, :, :, 512*2:512*3]
                save_image(B_img_save, output_path3, nrow=1, padding=0)


            img_res = Image.open(output_path).convert('RGB')
            err = mse(img_res, img_target)
            sum_err += err


            # print  ["output error", str(err) ]


            # print  output_path

        avg_err = sum_err / count
        # if avg_err > 0:
        #     print "avg_err: "
        #     print  avg_err
        #
        if avg_err < 1:
            continue

        output_txt_path = output_path_base + "error_" + str(avg_err) + ".txt"
        text_file = open(output_txt_path, "w")

        text_file.write(str(avg_err))
        text_file.close()

        sum_sum_err += avg_err


    avg_err = sum_sum_err / len(selected_names)
    # if avg_err > 0:
    #     print "final avg_err: "
    #     print  avg_err

    output_txt_path = output_root + opt.name + "/" + "error_" + str(avg_err) + ".txt"
    text_file = open(output_txt_path, "w")

    text_file.write(str(avg_err))
    text_file.close()


