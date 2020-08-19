import os
import cv2
import numpy as np
from tqdm import tqdm
import os.path as osp

path = "G:\\CV\\Result\\new\\challenge_task"
res_dir = "G:\\CV\\Result\\new_res"
dir_list = os.listdir(path)

for dir in dir_list:
    txt_name = dir[:-4] + '.txt'
    dir_path = osp.join(path, dir)
    obj_list = os.listdir(dir_path)

    txt_path = osp.join(res_dir, txt_name)
    with open(txt_path, 'w') as f:
        for info in obj_list:
            # print(info)
            f.writelines(info[:-4] + '\n')