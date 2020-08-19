"""
use rondom array replace objce witch was neglected
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.switch_backend('agg')
# plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['font.sans-serif'] = ['SimHei']
id2cls = {1: "行人", 2: "坐着人", 3: "自行车", 4: "汽车", 5: "面包车", 6: "货车", 7: "敞篷三轮", 8: "三轮", 9: "公交", 10: "轻骑"}


def region_enlarge(region, mask_shape, weight):
    """
    Args:
        mask_box: list of box
        image_size: (width, hight)
        ratio: int
    """
    width, hight = mask_shape
    rgn_w, rgn_h = region[2] - region[0], region[3] - region[1]
    center_x, center_y = region[0] + rgn_w / 2.0, region[1] + rgn_h / 2.0
    chip_area = rgn_w * rgn_h
    rect = np.sqrt(chip_area * weight)
    if max(rgn_w, rgn_h) <= rect:
        half_w = 0.5 * rect
        half_h = 0.5 * rect
    elif rgn_w > rect:
        half_w = 0.5 * rgn_w
        half_h = 0.5 * chip_area * weight / rgn_w
    else:
        half_h = 0.5 * rgn_h
        half_w = 0.5 * chip_area * weight / rgn_h
    half_w = min(half_w, width/2.0)
    half_h = min(half_h, hight/2.0)

    center_x = half_w if center_x < half_w else center_x
    center_y = half_h if center_y < half_h else center_y
    center_x = width - half_w if center_x > width - half_w else center_x
    center_y = hight - half_h if center_y > hight - half_h else center_y

    new_box = [center_x - half_w if center_x - half_w > 0 else 0,
               center_y - half_h if center_y - half_h > 0 else 0,
               center_x + half_w if center_x + half_w < width else width,
               center_y + half_h if center_y + half_h < hight else hight]

    return new_box


def show_image(img, cls, score, labels=None, img_name=None):
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams.update({'figure.max_open_warning': 0})
    plt.text(labels[0][0], labels[0][1], id2cls[cls] + ' | {:.2f}'.format(score), fontsize=15, color="b",)
    plt.imshow(img[..., ::-1])
    if labels is not None:
        labels = np.array(labels)
        plt.plot(labels[:, [0, 2, 2, 0, 0]].T, labels[:, [1, 1, 3, 3, 1]].T, '-', color='b', linewidth=1)
    plt.savefig(img_name)
    # plt.show()
    # ax.set_axis_off()
    plt.close()
    pass


img_path = "G:\\CV\\Dataset\\Detection\\Visdrone\\VisDrone2019-DET-test-challenge\\images"
img_list = os.listdir(img_path)
results = "G:\\CV\\Result\\cascade_Res\\results"
result_path = "G:\\CV\\Result\\challenge_task"
for ii, img_name in enumerate(tqdm(img_list)):
    # if ii < 80: continue
    img_file = osp.join(img_path, img_name)
    img = cv2.imread(img_file)
    img_h, img_w = img.shape[:2]
    anno_file = osp.join(results, img_name[:-4]+'.txt')
    img_res_dir = osp.join(result_path, img_name)
    if not osp.exists(img_res_dir):
        os.mkdir(img_res_dir)
    anno = 0
    annos = []
    with open(anno_file) as f:
        for line in f.readlines():
            box = np.array([int(v) for v in line.strip().split(',')[:4]])
            cls = int(line.strip().split(',')[-3])
            score = float(line.strip().split(',')[-4])
            box[2:] = box[:2] + box[2:]
            annos.append(box)
            new_box = np.array(region_enlarge(box, (img_w, img_h), 100)).astype(np.int)
            box = [box[0]-new_box[0], box[1]-new_box[1], box[2]-new_box[0], box[3]-new_box[1]]
            obj = img[new_box[1]:new_box[3], new_box[0]:new_box[2]]
            obj_name = line.strip() + '.jpg'
            obj_path = osp.join(img_res_dir, obj_name)
            if score >= 0.3:
                show_image(obj, cls, score, [box], obj_path)
            else:
                with open(obj_path[:-4] + '.txt', 'w') as f:
                    pass
            # cv2.imwrite(obj_path, obj)
            # show_image(obj)

    # show_image(img, annos, osp.join(img_res_dir, ))
