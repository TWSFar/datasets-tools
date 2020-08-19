import os
import cv2
import os.path as osp
import numpy as np


pool_dir = "G:\\CV\\Dataset\\Detection\\Visdrone\\TEMP\\Visdrone_ship"
# pool_dir = "G:\\CV\\Dataset\\Detection\Visdrone\TEMP\challenge\paster_pool_testchip"

file_list = os.listdir(pool_dir)

for old_name in file_list:
    info = old_name.split('_')
    if "副本" not in info[-1]:
        continue
    path = osp.join(pool_dir, old_name)
    # img = cv2.imread(path)
    # h, w = img.shape[:2]
    # ration = '{:.2f}'.format(w / h)
    # nonzero_pixel = (np.sum(img, axis=2) > 0).sum()
    # mb = img[..., 0].sum() / nonzero_pixel
    # mg = img[..., 1].sum() / nonzero_pixel
    # mr = img[..., 2].sum() / nonzero_pixel
    # bright = '{:.2f}'.format(0.3*mr + 0.6*mg + 0.1*mb)
    # temp = info[:-1] + ["c", info[-1][:-4], str(w), str(h), ration, bright+'.png']
    # new_name = "_".join(temp)
    # new_name = "_".join(info[:-2] + [bright, '.png'])
    new_name = "3_" + old_name[:-9] + '.jpg'
    new_path = osp.join(pool_dir, new_name)
    os.rename(path, new_path)
