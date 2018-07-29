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
# test_names.append("job_submit_22_multiD_unet512_patch128_vgg_lambda10")
# test_names.append("job_submit_23_multiD_unet512_patch256_vgg_lambda10")
#
# test_names.append("job_submit_32_multiD_unet512_patch256_vgg15")
# test_names.append("job_submit_43_multiD_unet1024_patch256_vgg_lambda10")
#
# test_names.append("job_submit_61_multiD_unet512_patch128old")
# test_names.append("job_submit_62_t15_multiD_unet1024")
# test_names.append("job_submit_82")
# test_names.append("job_submit_81")
#
#
#
# test_names.append("job_submit_93_render_to_real")
# test_names.append("job_submit_92_all_angel")
# # test_names.append("")
#
# test_names.append("job_submit_91_outdoor")




# test_names.append("job_submit_101_re1")
# test_names.append("job_submit_101C_re1")
#
# test_names.append("job_submit_103_re1")
# test_names.append("job_submit_105_re1_vgg")
#
# test_names.append("job_submit_102_re2")
# test_names.append("job_submit_104_re1")
#

# test_names.append("job_submit_101C_re1_pixel_tune")
test_names.append("job_submit_101C_re1_pixel")
# test_names.append("job_submit_101C_re1_pixel_2_tune")
test_names.append("job_submit_101C_re1_pixel_2")
test_names.append("job_submit_101C_re1_pixel_3")
test_names.append("job_submit_101C_re1_pixel_4")
test_names.append("job_submit_101C_re1_pixel_5")
test_names.append("job_submit_101C_re1_pixel_6")
test_names.append("job_submit_101C_re1_pixel_7")
# #
# test_names.append("job_submit_101C_re1_pixel_6_tune")
# test_names.append("job_submit_101C_re1_pixel_3_tune")



# test_names.append("job_submit_101C_re1_pixel_6")
# test_names.append("job_submit_105_re1_vgg")
# test_names.append("job_submit_105_re1_vgg_tune")


local_root = "./checkpoints/"

# local_root = "/home/dragon/00download_cluster/download_checkpoints/"

# remote_root = "wu@submit.unibe.ch:/home/ubelix/inf/wu/GPU/13-7_multiD_patch/checkpoints/"
remote_root = "wu@submit.unibe.ch:/home/ubelix/inf/wu/GPU/13-7_multiD_patch/checkpoints/"




for name in test_names:

    if not os.path.exists(local_root + name):
        os.mkdir(local_root + name)

    local_path = local_root + name + "/save_model/"

    if not os.path.exists(local_path):
        os.mkdir(local_path)

    remote_path = remote_root + name + "/save_model/latest_*"

    p = subprocess.Popen(["scp", "-r", remote_path, local_path])
    sts = os.waitpid(p.pid, 0)




# import datetime
#
# now = datetime.datetime.now()

# for name in test_names:
#
#     local_path = local_root + name + "/" + str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + "/"
#
#     if not os.path.exists(local_path):
#         os.mkdir(local_path)
#
#     remote_path = remote_root + name + "/web/images"
#
#     p = subprocess.Popen(["rsync", "-razSP", "--remove-source-files" ,remote_path, local_path])
#     sts = os.waitpid(p.pid, 0)
#
#     # p = subprocess.Popen(["scp", "-r", remote_path, local_path])
#     # sts = os.waitpid(p.pid, 0)
#     #
#     # p2 = subprocess.Popen(["rm", "-r", remote_path + "/*"])
#     # sts = os.waitpid(p2.pid, 0)












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




