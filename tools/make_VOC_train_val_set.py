import os
import os.path as osp
import random as rd
import shutil

root_dir = "G:\\CV\\Dataset\\Detection\\UnderWater\\UnderWater_VOC"
img_path = "G:\\CV\\Dataset\\Detection\\UnderWater\\train\\image"
label_path = "G:\\CV\\Dataset\\Detection\\UnderWater\\train\\box"

obj_dir = ["JPEGImages", "Annotations", "ImageSets/Main"]
for dirname in obj_dir:
    dir_path = osp.join(root_dir, dirname)
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

img_list = os.listdir(img_path)
rd.shuffle(img_list)
len_val = round(0.1*len(img_list))


img_path_voc = osp.join(root_dir, obj_dir[0])
label_path_voc = osp.join(root_dir, obj_dir[1])
set_path_voc = osp.join(root_dir, obj_dir[2])
trian_set = open(osp.join(set_path_voc, "train.txt"), 'w')
val_set = open(osp.join(set_path_voc, "val.txt"), 'w')

for i, filename in enumerate(img_list):
    img_id = osp.splitext(filename)[0]
    img_file = osp.join(img_path, filename)
    label_file = osp.join(label_path, img_id+'.xml')

    assert osp.isfile(img_file) and osp.isfile(label_file)

    if i < len_val:
        val_set.writelines(img_id + '\n')
    else:
        trian_set.writelines(img_id + '\n')
    shutil.move(img_file, img_path_voc)
    shutil.move(label_file, label_path_voc)

trian_set.close()
val_set.close()
