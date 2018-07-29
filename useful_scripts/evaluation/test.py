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


# opt.no_dropout = True
# opt.name = "cycle_angle_univer_512_unet"
# opt.which_epoch = "20"
# opt.which_model_netG = "unet_512"


opt.name = "univer_pix2pix_resnet_512"
opt.which_epoch = "latest"
opt.which_model_netG = "resnet_9blocks"
opt.model = "test"
opt.dataset_mode = 'single'
opt.norm = "batch"

opt.checkpoints_dir = "/home/dragon/00download_cluster/download_checkpoints/"

model = create_model(opt)

# input_root = "../selected_rnn_test_data/"

input_root = "/media/dragon/My Passport/output_test_new/"

output_root = "../../../00rnn_test_output/select_output_10_more/"



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


selected_names = [1004,  1006,  1011,  1017,  1018,  1025,  1029,  1041,  1121, 1277,
                  1014, 1096, 1122, 1164, 1239, 1297, 1309, 1354, 1377, 1410, 1441, 1469,
                  1041, 1099, 1137, 1187, 1250, 1306, 1322, 1356, 1392, 1422, 1444, 1470,
                  1055, 1101, 1154, 1207, 1260, 1308, 1323, 1358, 1407, 1427, 1445, 1484, 1480, 1477, 1473         ]

# selected_names = [1006, 1017, 1041, 1164, 1187, 1306, 1514, 5007, 1543, 1554, 1566, 1623, 1630, 1660, 1698, 1703, 1707,
#                   1711, 1720, 1724, 1726, 1728, 1737, 1758, 1813, 1819, 1879, 1913, 1941, 1961, 2221, 2240, 2291, 2316,
#                   3003, 3118, 3141, 3261, 3267, 3307, 3322, 3361, 3409, 3438, 3446, 3449, 3455, 3472, 3479, 3508, 3529,
#                   3533, 3584, 3635, 3646, 3686, 3688, 3708, 3720, 3778, 3795, 3814, 3869, 3881, 3926, 5009, 5015, 5027,
#                   5044, 5061, 5065, 5072, 5078]

# selected_names = [1006, 1017, 1041, 1164, 1187, 1306]


# for test_id in selected_names:
for test_id in range(3500, 4000):

    sum_err = 0

    output_path_base = output_root + opt.name + "/" + str(opt.which_epoch) + "/" + str(test_id) + "/"

    print  test_id

    for cam_id in range(0, 11):

        input_path = input_root + str(test_id) + "/" + "img-" + str(test_id) + "-cam-" + str(cam_id) + "-pair-1.png"

        if not os.path.exists(input_path):
            continue

        output_path_base = output_root + opt.name + "/" + str(opt.which_epoch) + "/" + str(test_id) + "/"

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


        # B = model.netG_A(A)
        B = model.netG(A)

        save_image(denorm(B.data), output_path, nrow=1, padding=0)

        # print  output_path


        target_path = input_root + str(test_id) + "/" + "img-" + str(test_id) + "-cam-" + str(cam_id) + "-pair-0.png"
        img_target = Image.open(target_path).convert('RGB')

        img_res = Image.open(output_path).convert('RGB')
        err = mse(img_res, img_target)
        sum_err += err





    avg_err = sum_err / 15
    if avg_err > 0:
        print "avg_err: "
        print  avg_err

    if avg_err < 1:
        continue

    output_txt_path = output_path_base + "error_" + str(avg_err) + ".txt"
    text_file = open(output_txt_path, "w")

    text_file.write(str(avg_err))
    text_file.close()

    sum_sum_err += avg_err

avg_err = sum_sum_err / len(selected_names)
if avg_err > 0:
    print "final avg_err: "
    print  avg_err

output_txt_path = output_root + opt.name + "/" + str(opt.which_epoch) + "/" + "error_" + str(avg_err) + ".txt"
text_file = open(output_txt_path, "w")

text_file.write(str(avg_err))
text_file.close()



# data_loader = CreateDataLoader(opt)
# dataset = data_loader.load_data()
# model = create_model(opt)
# visualizer = Visualizer(opt)
# create website
# web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
# webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
# for i, data in enumerate(dataset):
#     if i >= opt.how_many:
#         break
#     model.set_input(data)
#     model.test()
#     visuals = model.get_current_visuals_test()
#     img_path = model.get_image_paths()
#     print('process image... %s' % img_path)
#     visualizer.save_images(webpage, visuals, img_path)

# webpage.save()
