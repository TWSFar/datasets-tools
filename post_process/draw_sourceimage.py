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
box_colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
              (0, 255, 255), (255, 255, 255),
              (125, 125, 125), (0, 0, 0),
              (0, 125, 255), (0.439, 0.502, 0.412),
              (0, 0, 0)  # others
              )


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


def plot_img(img, bboxes):
    # img = img.astype(np.float64) / 255.0 if img.max() > 1.0 else img
    for i, bbox in enumerate(bboxes):
        try:
            if -1 in bbox:
                continue
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            id = int(bbox[4])
            score = float(bbox[5])
            if score <= 0.3:
                continue
            label = str(id) + ": " + "{:.2f}".format(score)
            # plot
            box_color = box_colors[min(id-1, len(box_colors)-1)]
            text_color = (1, 1, 1)
            cv2.putText(img, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX, 0.5, text_color, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=box_color, thickness=1)

        except Exception as e:
            print(e)
            continue

    return img


img_path = "G:\\CV\\Dataset\\Detection\\Visdrone\\VisDrone2019-DET-test-challenge\\images"
img_list = os.listdir(img_path)
results = "G:\\CV\\Result\\cascade_Res\\results"
result_path = "G:\\CV\\Result\\pred_image"
for ii, img_name in enumerate(tqdm(img_list)):

    img_file = osp.join(img_path, img_name)
    img_res_dir = osp.join(result_path, img_name)

    img = cv2.imread(img_file)
    anno_file = osp.join(results, img_name[:-4]+'.txt')
    annos = []
    with open(anno_file) as f:
        for line in f.readlines():
            box = np.array([int(v) for v in line.strip().split(',')[:4]])
            box[2:] = box[:2] + box[2:]
            box = box.tolist()
            cls = line.strip().split(',')[5]
            score = float(line.strip().split(',')[4])

            annos.append(box + [cls, score])

    anno
    res = plot_img(img, annos)
    cv2.imwrite(img_res_dir, res)