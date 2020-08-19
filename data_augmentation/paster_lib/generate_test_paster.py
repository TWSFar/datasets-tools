import os
import cv2
import json
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
from shapely.geometry import box as sgbox

hyp = {
    'dataset': 'VisDrone2019',
    'img_type': '.jpg',
    "coco": [0, 1, 3],
    "mask_json": "G:\\CV\\Dataset\\Detection\\Visdrone\\TEMP\\instances_mask_cascade_test.json",
    'data_dir': 'G:\\CV\\Dataset\\Detection\\Visdrone\\TEMP\\challenge',
}
hyp['img_dir'] = osp.join(hyp['data_dir'], 'images')
hyp['result_dir'] = osp.join(hyp['data_dir'], 'paster_pool_test')
classes = ('pedestrian', 'person', 'bicycle', 'car', 'van',
           'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
results = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
coco2visdrone = {0: 0, 1: 2, 3: 9}


def iou_calc(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU


def getPredMask(img, box, contours):
    binary = np.zeros(img.shape[:2])
    cv2.drawContours(binary, np.array(contours), -1, 1, cv2.FILLED)
    binary = np.expand_dims(binary, 2).repeat(3, axis=2)
    img = img * binary
    mask_cls = img[box[1]:box[3], box[0]:box[2], :]
    return mask_cls.astype(np.uint8)


def show_image(img, labels=None):
    plt.figure(figsize=(10, 5))
    if labels is not None:
        labels = np.array(labels)
        plt.plot(labels[:, [0, 0, 2, 2, 0]].T, labels[:, [1, 3, 3,  1, 1]].T, '-')
    plt.imshow(img[..., ::-1])
    # plt.savefig('mask.jpg')
    plt.show()
    plt.close()


def mask_pools():
    if not osp.exists(hyp['result_dir']):
        os.makedirs(hyp['result_dir'])
    print("load mask json...")
    with open(hyp['mask_json']) as f:
        mask_dataset = json.load(f)
    print("loaded!")

    for file_name, mask_info in tqdm(mask_dataset.items()):
        # image info
        img_name = file_name[:-4] + hyp['img_type']  # image name
        img_path = osp.join(hyp['img_dir'], img_name)  # image path
        img = cv2.imread(img_path)
        # show_image(img, box_all)
        for i, box in enumerate(mask_info["bbox"]):
            score = mask_info["scores"][i]
            cls = mask_info["classes"][i]
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            try:
                if cls in hyp['coco'] and score > 0.8:  # and box_area > 40*40:
                    new_box = np.array(box, dtype=np.int)
                    nex_cls = coco2visdrone[cls]
                    mask_cls = getPredMask(img, new_box, mask_info["segmentation"][i])
                    mask_name = '_'.join([str(nex_cls), str(results[nex_cls]), "{:.2f}".format(score)]) + '.png'
                    path = osp.join(hyp['result_dir'], mask_name)
                    cv2.imwrite(path, mask_cls)
                    results[nex_cls] += 1
                    # show_image(mask_cls)
            except:
                print(img_name)
    print(results)


if __name__ == '__main__':
    mask_pools()
