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
# test_names.append("t31_unet1024_patch128_multiD_vgg_sgd")
# test_names.append("t35_unet512_patch128_multiD_vgg_sgd")
# test_names.append("t37_unet1024_patch256new_multiD_vgg_sgd")

# test_names.append("t38_unet1024_gd50")
# test_names.append("t39_unet1024_gd10")
# test_names.append("t40_unet1024_vgg0")
# test_names.append("t41_unet1024_gd0")

test_names.append("t43_reconv1")
test_names.append("t44_reconv2")
test_names.append("t40_unet1024_vgg0")
test_names.append("t41_unet1024_gd0")
test_names.append("t45_reconv1_pixel")

# test_names.append("cycle_angle_real5_2K_single")



local_root = "/home/dragon/00download_cluster/download_checkpoints/"
# remote_root = "wu@submit.unibe.ch:/home/ubelix/inf/wu/GPU/13-7_multiD_patch/checkpoints/"

remote_root = "wu@cluster.iam.unibe.ch:/data/cgg/shihao/13-7_multiD_patch/checkpoints/"

# remote_root = "wu@cluster.iam.unibe.ch:/data/cgg/shihao/cycle_gan/checkpoints/"



for name in test_names:

    local_path = local_root + name + "/save_model/"
    # local_path = local_root

    if not os.path.exists(local_root + name ):
        os.mkdir(local_root + name )

    if not os.path.exists(local_path):
        os.mkdir(local_path)

    remote_path = remote_root + name + "/save_model/latest_*"
    # remote_path = remote_root + name + "/latest_*"

    p = subprocess.Popen(["scp", "-r", remote_path, local_path])
    sts = os.waitpid(p.pid, 0)



# for name in test_names:
#
#     local_path = local_root + name + "/images/"
#     # local_path = local_root
#
#     if not os.path.exists(local_path):
#         os.mkdir(local_path)
#
#     remote_path = remote_root + name + "/web/"
#
#     p = subprocess.Popen(["scp", "-r", remote_path, local_path])
#     sts = os.waitpid(p.pid, 0)

















# localpath = "./checkpoints/"
# remotepath = "wu@submit.unibe.ch:/home/ubelix/inf/wu/GPU/13-7_multiD_patch/checkpoints/job_submit_22_multiD_unet512_patch128_vgg_lambda10/save_model/latest_*"
# # remotepath = "wu@submit.unibe.ch:/home/ubelix/inf/wu/GPU/13-7_multiD_patch/train.py"
#
# p = subprocess.Popen(["scp", "-r", remotepath, localpath])
# sts = os.waitpid(p.pid, 0)

# username = "wu"
# password = "1202001Wsh!"
# server = "submit.unibe.ch"
#
# localpath = "./checkpoints/"
# remotepath = "/home/ubelix/inf/wu/GPU/13-7_multiD_patch/checkpoints/job_submit_22_multiD_unet512_patch128_vgg_lambda10"
#
# ssh = paramiko.SSHClient()
# ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
# ssh.connect(server, username=username, password=password)
# sftp = ssh.open_sftp()
# sftp.put(remotepath, localpath)
# sftp.close()
# ssh.close()


# os.system("scp wu@submit.unibe.ch:/home/ubelix/inf/wu/GPU/13-7_multiD_patch/checkpoints/job_submit_22_multiD_unet512_patch128_vgg_lambda10 ./checkpoints")




