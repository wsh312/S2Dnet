import commands
import os
import subprocess
import sys
import shutil

rootDir = os.path.dirname(os.path.abspath(__file__))
list_dirs = os.walk(rootDir)
for root, dirs, files in list_dirs:
    for d in dirs:
        input_dir_name = os.path.join(root, d)
        if os.path.isdir(input_dir_name) and "_recons" in input_dir_name:
            print(input_dir_name)
            # output_dir_name = input_dir_name + "_recons"
            # shutil.rmtree(output_dir_name)
            # os.mkdir(output_dir_name)

            image_dir = input_dir_name
            scene_dir = input_dir_name

            check_name = scene_dir+"/smvs-S.ply"
            if (os.path.exists(check_name) is False):
                continue

            check_name = scene_dir+"/smvs-clean.ply"
            if (os.path.exists(check_name)):
                continue

            # if os.path.exists(scene_dir):
            #     shutil.rmtree(scene_dir)

            fssrecon = subprocess.Popen([os.path.join(rootDir, "fssrecon"), scene_dir+"/smvs-S.ply", scene_dir+"/smvs-surface.ply"])
            # fssrecon = subprocess.Popen([os.path.join(rootDir, "fssrecon"), scene_dir+"/smvs-B.ply", scene_dir+"/smvs-surface.ply"])
            fssrecon.wait()

            meshclean = subprocess.Popen([os.path.join(rootDir, "meshclean"), " -p10 ", scene_dir+"/smvs-surface.ply", scene_dir+"/smvs-clean.ply"])
            meshclean.wait()



            dmrecon = subprocess.Popen([os.path.join(rootDir, "dmrecon"), "--keep-conf", scene_dir])
            dmrecon.wait()

            scene2pset  = subprocess.Popen([os.path.join(rootDir, "scene2pset"), "-F2", scene_dir, scene_dir+"/pset-L2.ply"])
            scene2pset .wait()