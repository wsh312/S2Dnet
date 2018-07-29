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
# import scp
import os
import paramiko
import subprocess

test_names = []


# test_names.append("job_submit_91_outdoor")
# test_names.append("job_submit_92_all_angel")
test_names.append("t43_reconv1")
test_names.append("t44_reconv2")
test_names.append("t40_unet1024_vgg0")
test_names.append("t41_unet1024_gd0")
test_names.append("t45_reconv1_pixel")
test_names.append("t41_unet1024_gd0")

# test_names.append("")




# local_root = "./checkpoints/"

local_root = "/home/dragon/00download_cluster/download_checkpoints/"

# remote_root = "wu@submit.unibe.ch:/home/ubelix/inf/wu/GPU/13-7_multiD_patch/checkpoints/"
# remote_root = "wu@submit.unibe.ch:/home/ubelix/inf/wu/GPU/13-7_multiD_patch/checkpoints/"

remote_root = "wu@cluster.iam.unibe.ch:/data/cgg/shihao/13-7_multiD_patch/checkpoints/"


# remote_root = "wu@cluster.iam.unibe.ch:/data/cgg/shihao/14-21_wgan2/checkpoints/"
# test_names = []
# test_names.append("t46_wgan_reconv1_pixel")

import datetime

now = datetime.datetime.now()

for name in test_names:

    if not os.path.exists(local_root + name):
        os.mkdir(local_root + name)

    local_path = local_root + name + "/" + str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + "/"

    if not os.path.exists(local_path):
        os.mkdir(local_path)

    remote_path = remote_root + name + "/web/images"

    p = subprocess.Popen(["rsync", "-razSP", "--remove-source-files" ,remote_path, local_path])
    sts = os.waitpid(p.pid, 0)

    # p = subprocess.Popen(["scp", "-r", remote_path, local_path])
    # sts = os.waitpid(p.pid, 0)
    #
    # p2 = subprocess.Popen(["rm", "-r", remote_path + "/*"])
    # sts = os.waitpid(p2.pid, 0)





