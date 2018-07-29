
import os
import subprocess
import sys
from shutil import copyfile
from PIL import Image


rootDir = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(rootDir)

list_dirs = os.walk(rootDir)
for root, dirs, files in list_dirs:
    for d in dirs:
        file_name = os.path.join(root, d)
        if (file_name.find(".png") != -1):
            print os.path.join(root, d)
            target_name = file_name.replace(".png", ".jpg")
            im = Image.open(file_name)
            im.save(target_name)
            os.unlink(file_name)
    for f in files:
        file_name = os.path.join(root, f)
        if (file_name.find(".png") != -1):
            print os.path.join(root, f)
            target_name = file_name.replace(".png", ".jpg")
            # target_name = target_name.replace("view", "dragon")
            img = Image.open(file_name)
            rgbimg = img.convert('RGB')
            rgbimg.save(target_name, "JPEG")
            os.unlink(file_name)



# model_name = "xyzrgb_statuette"
#
# for camera_id in range(compute_index_start, compute_index_end):
#     src_image_name = parent_path + "/New rendering style/cluster_xyzrgb_statuette_ordered_fixed_specularity/output/" + str(model_name) + "-" + str(camera_id) + "-mtl-" + str(extract_mtl_id) + ".png"
#     im = Image.open(src_image_name)
#
#     target_name = rootDir + "/input_img/" + str(model_name) + "-" + str(camera_id) + "-mtl-" + str(extract_mtl_id) + ".jpg"
#     im.save(target_name)
