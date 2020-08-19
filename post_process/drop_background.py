"""
use rondom array replace objce witch was neglected
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import os.path as osp
import matplotlib.pyplot as plt


def show_image(img, labels=None):
    plt.figure(figsize=(10, 5))
    if labels is not None:
        labels = np.array(labels)
        plt.plot(labels[:, [0, 0, 2, 2, 0]].T, labels[:, [1, 3, 3,  1, 1]].T, '-')
    plt.imshow(img[..., ::-1])
    # plt.savefig('mask.jpg')
    plt.show()
    plt.close()



img_path = "G:\\CV\\Dataset\\Detection\\Visdrone\\VisDrone2019-DET-test-challenge\\images"
img_list = os.listdir(img_path)
results = "G:\CV\\Result\\results_33.2"
result_path = "G:\\CV\\Result\\challenge_task"
for img_name in tqdm(img_list):
    img_file = osp.join(img_path, img_name)
    img = cv2.imread(img_file)
    anno_file = osp.join(results, img_name[:-4]+'.txt')
    img_res_dir = osp.join(result_path, img_name)
    os.mkdir(img_res_dir)
    with open(anno_file) as f:
        for line in f.readlines():
            box = np.array([int(v) for v in line.strip().split(',')[:4]])
            box[2:] = box[:2] + box[2:]
            obj = img[box[1]:box[3], box[0]:box[2]]
            obj_name = line.strip() + '.jpg'
            obj_path = osp.join(img_res_dir, obj_name)
            # pass
            cv2.imwrite(obj_path, obj)
            # show_image(obj)
