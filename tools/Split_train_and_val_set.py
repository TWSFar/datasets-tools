import os
import os.path as osp
import random as rd
import shutil

train_path = "JPEGImages"
label_path = "Annotations"

obj_dir = ["images/train", "images/val", "label/train", "label/val"]

for dirname in obj_dir:
    if not osp.exists(dirname):
        os.makedirs(dirname)


data = os.listdir(train_path)

rd.shuffle(data)

len_val = round(0.3*len(data))


for i, filename in enumerate(data):
    fd = osp.join(train_path, filename)
    fl = osp.join(label_path, filename.split('.')[0]+'.xml')
    if i < len_val:
        shutil.move(fd, obj_dir[1])
        shutil.move(fl, obj_dir[3])
    else:
        shutil.move(fd, obj_dir[0])
        shutil.move(fl, obj_dir[2])
        