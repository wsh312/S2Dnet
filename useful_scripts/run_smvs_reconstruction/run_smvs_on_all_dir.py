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

        if os.path.isdir(input_dir_name) and "_recons" not in input_dir_name:
            print(input_dir_name)
            output_dir_name = input_dir_name + "_recons"

            if os.path.isdir(output_dir_name):
                continue

            #shutil.rmtree(output_dir_name)
            os.mkdir(output_dir_name)
            #
            image_dir = input_dir_name
            scene_dir = output_dir_name
            #
            if os.path.exists(scene_dir):
                shutil.rmtree(scene_dir)

            makescene = subprocess.Popen([os.path.join(rootDir, "makescene"), "-i", image_dir, scene_dir])
            makescene.wait()


            # sfmrecon = subprocess.Popen([os.path.join(rootDir, "sfmrecon"), "--initial-pair=20,21", scene_dir])
            sfmrecon = subprocess.Popen([os.path.join(rootDir, "sfmrecon"),  scene_dir])
            sfmrecon.wait()

            smvsrecon = subprocess.Popen([os.path.join(rootDir, "smvsrecon"), "-S", scene_dir])
            # smvsrecon = subprocess.Popen([os.path.join(rootDir, "smvsrecon"), scene_dir])
            smvsrecon.wait()

            # fssrecon = subprocess.Popen([os.path.join(rootDir, "smvsrecon"), scene_dir+"/smvs-[B,S].ply", scene_dir+"/smvs-surface.ply"])
            # # fssrecon = subprocess.Popen([os.path.join(rootDir, "fssrecon"), scene_dir+"/smvs-B.ply", scene_dir+"/smvs-surface.ply"])
            # fssrecon.wait()
            #
            # meshclean = subprocess.Popen([os.path.join(rootDir, "meshclean"), " -p10 ", scene_dir+"/smvs-surface.ply", scene_dir+"/smvs-clean.ply"])
            # meshclean.wait()
            #
            # dmrecon = subprocess.Popen([os.path.join(rootDir, "dmrecon"), "--keep-conf", scene_dir])
            # dmrecon.wait()
            #
            # scene2pset  = subprocess.Popen([os.path.join(rootDir, "scene2pset"), "-F2", scene_dir, scene_dir+"/pset-L2.ply"])
            # scene2pset .wait()