import os
import cv2
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict

hyp = {
    'dataset': 'VisDrone2019',
    'img_type': '.jpg',
    'data_dir': 'G:\\CV\\Dataset\\Detection\\Visdrone\\TEMP',
}
hyp['txt_dir'] = osp.join(hyp['data_dir'], 'annotations')
hyp['img_dir'] = osp.join(hyp['data_dir'], 'images')
hyp['cls_pools'] = osp.join(hyp['data_dir'], 'cls_pools')


classes = ('pedestrian', 'person', 'bicycle', 'car', 'van',
           'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
rare = [2, 5, 6, 7, 8]
min_area = {1: 0.00051, 2: 0.00066, 4: 0.0025, 5: 0.00534587, 6: 0.001657, 7: 0.0017394, 8: 0.00509, 9: 0.00074}
results = {1: 0, 2: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
weight = {1: 1.23, 2: 1.087, 4: 1.208, 5: 1.3, 6: 1.04, 7: 1.027, 8: 1.07, 9: 1.247}

def iou_calc1(boxes1, boxes2):
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


def show_image(img, labels=None):
    plt.figure(figsize=(10, 5))
    if labels is not None:
        labels = np.array(labels)
        plt.plot(labels[:, [0, 0, 2, 2, 0]].T, labels[:, [1, 3, 3,  1, 1]].T, '-')
    plt.imshow(img[..., ::-1])
    # plt.savefig('mask.jpg')
    plt.show()
    plt.close()


def class_pool():
    if not osp.exists(hyp['cls_pools']):
        os.makedirs(hyp['cls_pools'])
    rare_list = []
    txt_files = os.listdir(hyp['txt_dir'])
    for id, file_name in enumerate(tqdm(txt_files)):
        # anno info
        anno_txt = os.path.join(hyp['txt_dir'], file_name)
        box_all, gt_cls = getGTBox(anno_txt)
        for rc in rare:
            if rc in gt_cls:
                rare_list.append(file_name)
                break
    """
        # image info
        img_name = file_name[:-4] + hyp['img_type']  # image name
        img_path = osp.join(hyp['img_dir'], img_name)  # image path
        img = cv2.imread(img_path)
        # show_image(img, box_all)
        for i, box in enumerate(box_all):
            areagt = (box[2] - box[0]) * (box[3] - box[1])
            area = areagt / (img.shape[0] * img.shape[1])
            if box[0] <= 10 or box[1] <= 10 or box[2] >= img.shape[1] - 10 or box[3] >= img.shape[0] - 10:
                continue
            if gt_cls[i] in rare and (iou_calc1(box, box_all) > 0).sum() <= 1 and (area > 5 * weight[gt_cls[i]] * min_area[gt_cls[i]] or areagt > weight[gt_cls[i]]*weight[gt_cls[i]]*50*50):
                box = np.array(box, dtype=np.int)
                pattern = img[box[1]:box[3], box[0]:box[2], :]
                path = osp.join(hyp['cls_pools'], str(gt_cls[i]) + "_" + str(results[gt_cls[i]])) + '.jpg'
                cv2.imwrite(path, pattern)
                results[gt_cls[i]] += 1
                # show_image(pattern)
    print(results)
    """
    with open("rare_list.txt", 'w') as f:
        for line in rare_list:
            f.writelines(line + '\n')


if __name__ == '__main__':
    class_pool()
