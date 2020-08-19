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
    "rare": [0, 1, 2, 6, 7, 9],
    "mask_json": "G:\\CV\\Dataset\\Detection\\Visdrone\\TEMP\\instances_mask_cascade.json",
    'data_dir': 'G:\\CV\\Dataset\\Detection\\Visdrone\\TEMP',
}
hyp['txt_dir'] = osp.join(hyp['data_dir'], 'annotations')
hyp['img_dir'] = osp.join(hyp['data_dir'], 'images')
hyp['result_dir'] = osp.join(hyp['data_dir'], 'paster_pool')
classes = ('pedestrian', 'person', 'bicycle', 'car', 'van',
           'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
results = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}


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


def getGTBox(anno_path, **kwargs):
    box_all = []
    gt_cls = []
    with open(anno_path, 'r') as f:
        data = [x.strip().split(',')[:8] for x in f.readlines()]
        annos = np.array(data)

    bboxes = annos[annos[:, 4] == '1'][:, :6].astype(np.float64)
    for bbox in bboxes:
        if bbox[2] <= 0 or bbox[3] <= 0:
            print(osp.basename(anno_path) + ' exist an illegal side:')
            print(bbox)
            print('illegal side has been abandoned')
            continue
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        box_all.append(bbox[:4].tolist())
        gt_cls.append(int(bbox[5]) - 1)  # index begin idx 0

    return box_all, gt_cls


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
    txt_files = os.listdir(hyp['txt_dir'])
    print("load mask json...")
    with open(hyp['mask_json']) as f:
        mask_dataset = json.load(f)
    print("loaded!")

    for id, file_name in enumerate(tqdm(txt_files)):
        # anno info
        anno_txt = os.path.join(hyp['txt_dir'], file_name)
        box_all, gt_cls = getGTBox(anno_txt)

        # mask info
        mask_info = mask_dataset[file_name[:-4] + '.jpg']

        # image info
        img_name = file_name[:-4] + hyp['img_type']  # image name
        img_path = osp.join(hyp['img_dir'], img_name)  # image path
        img = cv2.imread(img_path)
        # show_image(img, box_all)
        for i, box in enumerate(box_all):
            if (len(mask_info["bbox"]) == 0) or gt_cls[i] not in hyp['rare']:
                continue
            iou1 = iou_calc(box, box_all)
            iou2 = iou_calc(box, mask_info["bbox"])
            max_iou = max(iou2)
            if max_iou < 0.9:
                continue
            mask_id = iou2.argmax()
            max_box = mask_info["bbox"][mask_id]
            max_score = mask_info["scores"][mask_id]
            new_box = sgbox(max_box[0], max_box[1], max_box[2], max_box[3]).intersection(sgbox(box[0], box[1], box[2], box[3])).bounds
            new_box_area = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
            try:
                if sum(iou1 > 0.) <= 1 and max_score > 0.7 and new_box_area > 40*40:
                    new_box = np.array(new_box, dtype=np.int)
                    mask_cls = getPredMask(img, new_box, mask_info["segmentation"][iou2.argmax()])
                    mask_name = '_'.join([str(gt_cls[i]), str(results[gt_cls[i]]), "{:.2f}".format(max_iou), "{:.2f}".format(max_score)]) + '.png'
                    path = osp.join(hyp['result_dir'], mask_name)
                    cv2.imwrite(path, mask_cls)
                    results[gt_cls[i]] += 1
                    # show_image(mask_cls)
            except:
                print(img_name)
    print(results)


if __name__ == '__main__':
    mask_pools()
