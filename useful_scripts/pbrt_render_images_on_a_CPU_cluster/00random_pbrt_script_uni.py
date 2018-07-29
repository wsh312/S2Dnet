import os
import subprocess
import sys

import math
import numpy as np
import random
import time
from math import fabs
from shutil import copyfile
#import Image
#import ImageOps



first_img_id = int(sys.argv[1])
last_img_id = int(sys.argv[2])

img_size = 512

rootDir = os.path.dirname(os.path.abspath(__file__))
model_root_dir = rootDir + "/geometry/"

model_file_names = []
for (dirpath, dirnames, filenames) in os.walk(model_root_dir):
    model_file_names.extend(filenames)
    break
model_num = len(model_file_names)

# use randomly generated camer positions
camera_positions = []
f = open('sphere_view_points_1k_perspective_700_wlop_ordered.xyz', 'r')
for line in f:
    row = line.split()
    camera_positions.append(row)

camera_views_num = len(camera_positions)


directional_positions = []
f = open('directional_light_positions.txt', 'r')
for line in f:
    row = line.split()
    directional_positions.append(row)

directional_num = len(directional_positions)
f.close()
#random.seed(5)


f=open("rgb.txt","r")
lines=f.readlines()
r_vals =[]
g_vals =[]
b_vals =[]

for x in lines:
    r_vals.append(x.split(' ')[0])
    g_vals.append(x.split(' ')[1])
    b_vals.append(x.split(' ')[2])
f.close()
color_num = len(r_vals)




min_roughness = 0
max_roughness = 0.016  # 0.02

plate_z_value = -1.3

pair_id0 = 0
pair_id1 = 2

for img_id in range(first_img_id, last_img_id):
    # camera_id = int(random.random()*camera_views_num)
    rand_camera_id = random.randrange(camera_views_num)
    if rand_camera_id <= 1 or rand_camera_id >= (camera_views_num -2):
        continue

    rand_camera_ids = []
    rand_camera_ids.append(rand_camera_id)
    rand_camera_ids.append(rand_camera_id - 1)
    rand_camera_ids.append(rand_camera_id - 2)
    rand_camera_ids.append(rand_camera_id + 1)
    rand_camera_ids.append(rand_camera_id + 2)

    rand_roughness = random.uniform(min_roughness, max_roughness)


    rand_model_id = random.randrange(model_num)
    rand_model_name = model_file_names[rand_model_id]
    rand_model_name = rand_model_name.replace("_aligned.ply", "")

    rand_obj_rotate = random.uniform(0, 360)
    line_rotate = "Rotate " + str(rand_obj_rotate) + " 0 0 1 " + '\n'

    rand_color_id = random.randrange(color_num)
    color_str = str(round(float(r_vals[rand_color_id]) / 255., 3)) + " " + str(
        round(float(g_vals[rand_color_id]) / 255., 3)) + " " + str(round(float(b_vals[rand_color_id]) / 255., 3))
    str_k_color = " \"color k\" [" + color_str + "] "

    rand_color_id = random.randrange(color_num)

    color_str = str(round(float(r_vals[rand_color_id]) / 255., 3)) + " " + str(
        round(float(g_vals[rand_color_id]) / 255., 3)) + " " + str(round(float(b_vals[rand_color_id]) / 255., 3))
    str_eta_color = " \"color eta\" [" + color_str + "] "

    line_shape = "Shape \"plymesh\" \"string filename\"  \"geometry/" + str(rand_model_name) + "_aligned.ply\"" + '\n' + "AttributeEnd" + '\n' + '\n'


    info_name = rootDir + "/output_info/" + "img-" + str(img_id) + "_info.txt"
    f_info = open(info_name, 'w+')
    f_info.write(rand_model_name + "_aligned.ply")
    f_info.close()

    line_attri_bigin = "AttributeBegin" + '\n'

    diffuse_roughness = 0.16
    line_matetrial_0_candidates = []
    str_al = "\"spectrum k\" \"spds/Al.k.spd\" \"spectrum eta\" \"spds/Al.eta.spd\""  # silver
    str_au = "\"spectrum k\" \"spds/Au.k.spd\" \"spectrum eta\" \"spds/Au.eta.spd\""  # gold
    str_ag = "\"spectrum k\" \"spds/Au.k.spd\" \"spectrum eta\" \"spds/Ag.eta.spd\""  # silver_gold
    line_matetrial_0_silver = "Material \"metal\" " + str_al + "\"float roughness\" [" + str(
        diffuse_roughness) + "] " + '\n'
    line_matetrial_0_gold = "Material \"metal\" " + str_au + " \"float roughness\" [" + str(
        diffuse_roughness) + "] " + '\n'
    line_matetrial_0_copper = "Material \"metal\" " + "" + " \"float roughness\" [" + str(
        diffuse_roughness) + "] " + '\n'
    line_matetrial_0_silver_gold = "Material \"metal\" " + str_ag + " \"float roughness\" [" + str(
        diffuse_roughness) + "] " + '\n'
    line_matetrial_0_rand_color = "Material \"metal\" " + str_k_color + str_eta_color + " \"float roughness\" [" + str(diffuse_roughness) + "] " + '\n'
    line_material_0_matte = "Material \"matte\" "
    
    line_matetrial_0_candidates = [line_material_0_matte]


    line_matetrial_1_candidates = []
    str_al = "\"spectrum k\" \"spds/Al.k.spd\" \"spectrum eta\" \"spds/Al.eta.spd\" "  # silver
    str_au = "\"spectrum k\" \"spds/Au.k.spd\" \"spectrum eta\" \"spds/Au.eta.spd\" "  # gold
    str_ag = "\"spectrum k\" \"spds/Au.k.spd\" \"spectrum eta\" \"spds/Ag.eta.spd\" "  # silver_gold
    line_matetrial_1_silver = "Material \"metal\" " + str_al + " \"float roughness\" [" + str(
        rand_roughness) + "] " + '\n'
    line_matetrial_1_gold = "Material \"metal\" " + str_au + " \"float roughness\" [" + str(
        rand_roughness) + "] " + '\n'
    line_matetrial_1_copper = "Material \"metal\" " + " " + " \"float roughness\" [" + str(
        rand_roughness) + "] " + '\n'
    line_matetrial_1_silver_gold = "Material \"metal\" " + str_ag + " \"float roughness\" [" + str(
        rand_roughness) + "] " + '\n'
    line_matetrial_1_rand_color = "Material \"metal\" " + str_eta_color + " \"float roughness\" [" + str(
        rand_roughness) + "] " + '\n'

    line_matetrial_1_candidates = [line_matetrial_1_gold, line_matetrial_1_silver, line_matetrial_1_copper, line_matetrial_1_silver_gold, line_matetrial_1_copper, line_matetrial_1_gold, line_matetrial_1_silver, line_matetrial_1_copper, line_matetrial_1_silver_gold, line_matetrial_1_copper, line_matetrial_1_gold, line_matetrial_1_silver, line_matetrial_1_copper, line_matetrial_1_silver_gold, line_matetrial_1_copper,line_matetrial_1_rand_color, line_matetrial_1_gold, line_matetrial_1_gold, line_matetrial_1_gold]


    rand_material_id = random.randrange(len(line_matetrial_1_candidates))
    line_material_0 = line_matetrial_0_candidates[0]
    line_material_1 = line_matetrial_1_candidates[rand_material_id]


    materials = []
    materials.append(line_material_0)
    materials.append(line_material_1)




    # generate different lighting examples
    infinite_candidates = []
    light_20060807_wells6_hd = "AttributeBegin" + '\n' + "" + "LightSource \"infinite\"  \"string mapname\" [\"textures/20060807_wells6_hd.exr\"] \"color scale\" [2.1 2.1 2.1]" + '\n' + "AttributeEnd" + '\n'
    light_sky_day = "AttributeBegin" + '\n' + "LightSource \"infinite\"  \"string mapname\" [\"textures/sky-day.exr\"]  \"color scale\" [1.3 1.3 1.3]" + '\n' + "AttributeEnd" + '\n'
    light_sky = "AttributeBegin" + '\n' + "LightSource \"infinite\"  \"string mapname\" [\"textures/sky.exr\"]  \"color scale\" [2.1 2.1 2.1]" + '\n' + "AttributeEnd" + '\n'
    light_sky_dusk = "AttributeBegin" + '\n' + "LightSource \"infinite\"  \"string mapname\" [\"textures/skylight-dusk.exr\"]  \"color scale\" [1.5 1.5 1.5]" + '\n' + "AttributeEnd" + '\n'
    light_sky_morn = "AttributeBegin" + '\n' + "LightSource \"infinite\"  \"string mapname\" [\"textures/skylight-morn.exr\"]  \"color scale\" [1.3 1.3 1.3]" + '\n' + "AttributeEnd" + '\n'
    light_Alexs_Apt = "AttributeBegin" + '\n' + "LightSource \"infinite\"  \"string mapname\" [\"textures/Alexs_Apt_2k.exr\"]  \"color scale\" [2.4 2.4 2.4]" + '\n' + "AttributeEnd" + '\n'
    light_Chelsea_Stairs = "AttributeBegin" + '\n' + "LightSource \"infinite\"  \"string mapname\" [\"textures/Chelsea_Stairs_3k.exr\"]  \"color scale\" [2.5 2.5 2.5]" + '\n' + "AttributeEnd" + '\n'
    light_hdrvfx_zanla = "AttributeBegin" + '\n' + "LightSource \"infinite\"  \"string mapname\" [\"textures/hdrvfx_zanla.exr\"]  \"color scale\" [2.4 2.4 2.4]" + '\n' + "AttributeEnd" + '\n'
    light_Cave_Room = "AttributeBegin" + '\n' + "LightSource \"infinite\"  \"string mapname\" [\"textures/Mt-Washington-Cave-Room_Ref.exr\"]  \"color scale\" [2.4 2.4 2.4]" + '\n' + "AttributeEnd" + '\n'
    light_Gold_Room = "AttributeBegin" + '\n' + "LightSource \"infinite\"  \"string mapname\" [\"textures/Mt-Washington-Gold-Room_Ref.exr\"]  \"color scale\" [2.3 2.3 2.3]" + '\n' + "AttributeEnd" + '\n'
    light_Newport_Loft = "AttributeBegin" + '\n' + "LightSource \"infinite\"  \"string mapname\" [\"textures/Newport_Loft_Ref.exr\"]  \"color scale\" [2.6 2.6 2.6]" + '\n' + "AttributeEnd" + '\n'
    light_ProvWash = "AttributeBegin" + '\n' + "LightSource \"infinite\"  \"string mapname\" [\"textures/ProvWash_Ref.exr\"]  \"color scale\" [2.4 2.4 2.4]" + '\n' + "AttributeEnd" + '\n'
    light_Tokyo_BigSight = "AttributeBegin" + '\n' + "LightSource \"infinite\"  \"string mapname\" [\"textures/Tokyo_BigSight_3k.exr\"]  \"color scale\" [2.4 2.4 2.4]" + '\n' + "AttributeEnd" + '\n'
    light_WoodenDoor_Ref = "AttributeBegin" + '\n' + "LightSource \"infinite\"  \"string mapname\" [\"textures/WoodenDoor_Ref.exr\"]  \"color scale\" [1.6 1.6 1.6]" + '\n' + "AttributeEnd" + '\n'
    light_ = "AttributeBegin" + '\n' + "LightSource \"infinite\"  \"string mapname\" [\"textures/.exr\"]  \"color scale\" [2.4 2.4 2.4]" + '\n' + "AttributeEnd" + '\n'

    infinite_candidates = [light_20060807_wells6_hd, light_20060807_wells6_hd, light_Chelsea_Stairs, light_hdrvfx_zanla, light_Cave_Room, light_sky,
                           light_Gold_Room, light_Newport_Loft, light_ProvWash, light_Tokyo_BigSight,
                           light_WoodenDoor_Ref]
    line_light_infinite = random.choice(infinite_candidates)


    camera_noises = []
    for noise_id in range(len(rand_camera_ids)):
        camera_noise = (np.random.rand(3) - .5) * 2
        camera_noise *= 0.2555

        camera_noises.append(camera_noise)

    # generate random displacement for the light direction (just move a bit away from the camera direction)
    direct_noise = (np.random.rand(3) - .5) * 2
    direct_noise *= 8.5

    lamp_noise1 = (np.random.rand(3) - .5) * 2
    lamp_noise1 *= 5
    lamp_noise2 = (np.random.rand(3) - .5) * 2
    lamp_noise2 *= 5
            
    for pair_id in range(pair_id0, pair_id1):
        for cam_id in range(len(rand_camera_ids)):
            pbrt_file_name = "img-" + str(img_id) + "-pair-" + str(pair_id) + "-cam-" + str(cam_id)  + ".pbrt"

            check_name = rootDir + "/output_color/" + "img-" + str(img_id) + "-pair-" + str(pair_id) + "-cam-" + str(cam_id)  + ".png"
            if(os.path.exists(check_name)):
               print(["file exist: ", check_name])
               continue

            f = open(pbrt_file_name, "w")

            line_film = "Film \"image\" \"integer xresolution\" [" + str(img_size) + "] \"integer yresolution\" [" + str(img_size) + "]" + '\n' + "\"string filename\"  \"" + "img-" + str(img_id) + "-pair-" + str(pair_id) + "-cam-" + str(cam_id)  +  ".exr\"" + '\n'

            rand_camera_id = rand_camera_ids[cam_id]
            camera_noise = camera_noises[cam_id]
            line_camera = "LookAt " + str(float(camera_positions[rand_camera_id][0]) + camera_noise[0]) + " " + str(float(camera_positions[rand_camera_id][1]) + camera_noise[1]) + " " + str(fabs(float(camera_positions[rand_camera_id][2]) + camera_noise[2])) + " " + " 0 0 0 0 0 1" + '\n' "Camera \"perspective\" \"float fov\" [33] " + '\n'

            line3_1 = "Sampler \"halton\" \"integer pixelsamples\" [8]" + '\n'
            line3_2 = "Sampler \"halton\" \"integer pixelsamples\" [32]" + '\n'
            line3_3 = "Sampler \"halton\" \"integer pixelsamples\" [2048]" + '\n'
            if pair_id == 0:
                line3_3 = "Sampler \"halton\" \"integer pixelsamples\" [1024]" + '\n'



            line_integrator = "Integrator \"path\" \"integer maxdepth\" 3" + '\n'
            line_begin = '\n' + "WorldBegin" + '\n'


            line_integrator_0 = "Integrator \"whitted\" " + '\n'
            line_light_infinite_0 = ""
            directional_num = len(directional_positions)
            
            
            line_light_directional_0 = "AttributeBegin" + '\n'
            light_source = np.zeros(3)
            light_source[0] = float(camera_positions[rand_camera_id][0]) + camera_noise[0]
            light_source[1] = float(camera_positions[rand_camera_id][1]) + camera_noise[1]
            light_source[2] = fabs(float(camera_positions[rand_camera_id][2]) + camera_noise[2])
            line_light_directional_0 += "LightSource \"distant\" \"color scale\" [4.5 4.5 4.5] \"point from\" [" + str(light_source[0]) + " " + str(light_source[1]) + " " + str(light_source[2]) + "] \"point to\" [0 0 0] " + '\n'
            line_light_directional_0 += "AttributeEnd" + '\n'

            line_light_directional_1 = "AttributeBegin" + '\n'
            line_light_directional_1 += "LightSource \"distant\" \"color scale\" [1.15 1.15 1.15] \"point from\" [" + str(light_source[0]) + " " + str(light_source[1]) + " " + str(light_source[2]) + "] \"point to\" [0 0 0] " + '\n'
            line_light_directional_1 += "AttributeEnd" + '\n'
            
            line_lamp1 = "AttributeBegin Translate " + str(lamp_noise1[0]*1.5) + " " + str(lamp_noise1[1]*1.5) + " " + str(lamp_noise1[2] + 4.) + " AreaLightSource \"diffuse\" \"rgb L\" [ 1.8 1.8 1.8 ]  Shape \"sphere\" \"float radius\" .25  AttributeEnd" + '\n'


            line_lamp2 = "AttributeBegin  Translate " + str(lamp_noise2[0]*1.5) + " " + str(lamp_noise2[1]*1.5) + " " + str(lamp_noise1[2] + 4.) + " AreaLightSource \"diffuse\" \"rgb L\" [ 1.8 1.8 1.8 ]  Shape \"sphere\" \"float radius\" .25 AttributeEnd" + '\n'

            line_floor = "AttributeBegin" + '\n' + "Material \"plastic\" \"color Kd\" [.1 .1 .1] \"color Ks\" [.7 .7 .7] \"float roughness\" .1 " + '\n' + "Translate 0 0 " + str(plate_z_value) + " Shape \"trianglemesh\" \"point P\" [ -1000 -1000 0   1000 -1000 0   1000 1000 0 -1000 1000 0 ] \"float uv\" [ 0 0 1 0 1 1 0 1 ] \"integer indices\" [ 0 1 2 2 3 0]" + '\n' + "AttributeEnd" + '\n'
            # line6 = "AttributeBegin" + '\n' + "Texture \"lines\" \"color\" \"imagemap\" \"string filename\" \"textures/lines6.exr\"" + '\n' +"Material \"matte\" \"color Kd\" [.1 .1 .1] \"texture Kd\" \"lines\" \"color Ks\" [.7 .7 .7] \"float roughness\" .1" + '\n' +   "Translate 0 0 -0.4423 Shape \"trianglemesh\" \"point P\" [ -10 -10 0   10 -10 0   10 10 0 -10 10 0 ] \"float uv\" [ 0 0 1 0 1 1 0 1 ] \"integer indices\" [ 0 1 2 2 3 0]" + '\n'  + "AttributeEnd"+ '\n'

            line_floor = ""
            line9 = ""
            line10 = ""
            line11 = "WorldEnd"

            if pair_id == 0:
                f.write(
                line_film + line_camera + line3_1 + line3_2 + line3_3 + line_integrator_0 + line_begin + line_light_infinite + line_light_directional_0 + line_floor + line_attri_bigin + line_rotate + materials[pair_id] + line_shape + line11)
            else:
                f.write(line_film + line_camera + line3_1 + line3_2 + line3_3 + line_integrator + line_begin + line_light_infinite + line_light_directional_1 + line_floor + line_attri_bigin + line_rotate + materials[pair_id] + line_shape + line11)
                #f.write(line_film + line_camera + line3_1 + line3_2 + line3_3 + line_integrator + line_begin + line_light_infinite + line_light_directional_1 + line_lamp1 + line_lamp2 + line_floor + line_attri_bigin + line_rotate + materials[pair_id] + line_shape + line11)
            f.close()



for img_id in range(first_img_id, last_img_id):
    for pair_id in range(pair_id0, pair_id1):
        for cam_id in range(len(rand_camera_ids)):
            pbrt_file_name = "img-" + str(img_id) + "-pair-" + str(pair_id) + "-cam-" + str(cam_id)  + ".pbrt"

            check_name = rootDir + "/output_color_multi_new/" + "img-" + str(img_id) + "-pair-" + str(pair_id) + "-cam-" + str(cam_id)  + ".png"
            if(os.path.exists(check_name)):
               print(["file exist: ", check_name])
               continue

            pbrt = subprocess.Popen([os.path.join(rootDir, "pbrt"), rootDir + "/" "img-" + str(img_id) + "-pair-" + str(pair_id) + "-cam-" + str(cam_id)  + ".pbrt"])
            pbrt.wait()
            exrt4 = subprocess.Popen(
                [os.path.join(rootDir, "exrtopng"),
                 rootDir + "/" + "img-" + str(img_id) + "-pair-" + str(pair_id) + "-cam-" + str(cam_id)  + ".exr",
                 rootDir + "/output_color_multi_new/" + "img-" + str(img_id) + "-pair-" + str(pair_id) + "-cam-" + str(cam_id)  + ".png"])
            exrt4.wait()


for img_id in range(first_img_id, last_img_id):
    for pair_id in range(pair_id0, pair_id1):
        for cam_id in range(len(rand_camera_ids)):
            pbrt_file_name = rootDir + "/" + "img-" + str(img_id) + "-pair-" + str(pair_id) + "-cam-" + str(cam_id)  + ".pbrt"
            output_pbrt_file_name =  rootDir + "/output_pbrt_file/" + "img-" + str(img_id) + "-pair-" + str(pair_id) + "-cam-" + str(cam_id)  + ".pbrt"
            if os.path.exists(pbrt_file_name):
                copyfile(pbrt_file_name, output_pbrt_file_name)
                os.remove(pbrt_file_name)

            exr_file_name = rootDir + "/" + "img-" + str(img_id) + "-pair-" + str(pair_id) + "-cam-" + str(cam_id)  + ".exr"
            output_exr_file_name =  rootDir + "/output_exr_file/" + "img-" + str(img_id) + "-pair-" + str(pair_id) + "-cam-" + str(cam_id)  + ".exr"
            if os.path.exists(exr_file_name):
                copyfile(exr_file_name, output_exr_file_name)
                os.remove(exr_file_name)



#for img_id in range(first_img_id, last_img_id):
#
#    img0_path = rootDir + "/output_color/" + "img-" + str(img_id) + "-pair-" + str(0) + ".png"
#    img1_path = rootDir + "/output_color/" + "img-" + str(img_id) + "-pair-" + str(1) + ".png"
#    img_concat_path = rootDir + "/output_color_concat/" + "img-" + str(img_id) + "_concat.png"
#    img_concat_flip_path = rootDir + "/output_color_concat/" + "img-" + str(img_id) + "_concat_flip.png"
#
#    if os.path.exists(img_concat_path):
#        continue
#
#    img0 = Image.open(img0_path)
#    img1 = Image.open(img1_path)
#
#    img0_flip = ImageOps.mirror(img0)
#    img1_flip = ImageOps.mirror(img1)
#
#
#    img_concat = Image.new('RGB', (img_size*2, img_size))
#    img_concat.paste(img0, (0, 0))
#    img_concat.paste(img1, (img_size, 0))
#    img_concat.save(img_concat_path)
#
#    img_concat_flip = Image.new('RGB', (img_size*2, img_size))
#    img_concat_flip.paste(img0_flip, (0, 0))
#    img_concat_flip.paste(img1_flip, (img_size, 0))
#    img_concat_flip.save(img_concat_flip_path)
