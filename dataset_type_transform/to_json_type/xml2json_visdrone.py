import os
import json
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from collections import OrderedDict

hyp = {
    'dataset': 'VisDrone2019',
    'img_type': '.jpg',
    'mode': 'train',  # for save instance_train.json
    'data_dir': 'G:\\CV\\Dataset\\检测\\Visdrone\\VisDrone2019-DET-train',
}
hyp['json_dir'] = osp.join(hyp['data_dir'], 'annotations_json')
hyp['xml_dir'] = osp.join(hyp['data_dir'], 'annotations_xml')
hyp['img_dir'] = osp.join(hyp['data_dir'], 'images')
hyp['set_file'] = osp.join(hyp['data_dir'], 'ImageSet', 'Main', hyp['mode'] + '.txt')


class getItem(object):
    def __init__(self):
        self.classes = ('pedestrian', 'person', 'bicycle', 'car', 'van',
                        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
        # self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    def get_img_item(self, file_name, image_id, size):
        """Gets a image item."""
        image = OrderedDict()
        image['file_name'] = file_name
        image['height'] = int(size['height'])
        image['width'] = int(size['width'])
        image['id'] = image_id
        return image

    def get_ann_item(self, bbox, img_id, cat_id, anno_id):
        """Gets an annotation item."""
        x1 = bbox[0]
        y1 = bbox[1]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        annotation = OrderedDict()
        annotation['segmentation'] = [[x1, y1, x1, (y1 + h), (x1 + w), (y1 + h), (x1 + w), y1]]
        annotation['area'] = w * h
        annotation['iscrowd'] = 0
        annotation['image_id'] = img_id
        annotation['bbox'] = [x1, y1, w, h]
        annotation['category_id'] = cat_id
        annotation['id'] = anno_id
        return annotation

    def get_cat_item(self):
        """Gets an category item."""
        categories = []
        for idx, cat in enumerate(self.classes):
            cate = {}
            cate['supercategory'] = cat
            cate['name'] = cat
            cate['id'] = idx
            categories.append(cate)

        return categories


def getGTBox(anno_path, **kwargs):
    box_all = []
    gt_cls = []
    xml = ET.parse(anno_path).getroot()
    # y1, x1, y2, x2
    pts = ['ymin', 'xmin', 'ymax', 'xmax']
    # bounding boxes
    for obj in xml.iter('object'):
        bbox = obj.find('bndbox')
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text) - 1
            bndbox.append(cur_pt)
        box_all += [bndbox]
        cls = obj.find('name').text
        gt_cls.append(int(cls))

    return box_all, gt_cls


def make_json():
    item = getItem()
    images = []
    annotations = []
    anno_id = 0

    # categories
    categories = item.get_cat_item()

    with open(hyp['set_file'], 'r') as f:
        xml_list = f.readlines()
    for id, file_name in enumerate(tqdm(xml_list)):
        img_id = id

        # anno info
        anno_xml = os.path.join(hyp['xml_dir'], file_name + '.xml')
        box_all, gt_cls = getGTBox(anno_xml)
        for ii in range(len(box_all)):
            annotations.append(
                item.get_ann_item(box_all[ii], img_id, gt_cls[ii], anno_id))
            anno_id += 1

        # image info
        img_name = file_name[:-4] + hyp['img_type']  # image name
        img_path = osp.join(hyp['img_dir'], img_name)  # image path
        tsize = plt.imread(img_path).shape[:2]  # (h, w)
        size = {'height': tsize[0], 'width': tsize[1]}
        image = item.get_img_item(img_name, img_id, size)
        images.append(image)

    # all info
    ann = OrderedDict()
    ann['images'] = images
    ann['categories'] = categories
    ann['annotations'] = annotations

    # saver
    if not osp.exists(hyp['json_dir']):
        os.makedirs(hyp['json_dir'])
    save_file = os.path.join(hyp['json_dir'], 'instances_{}.json'.format(hyp['mode']))
    print('Saving annotations to {}'.format(save_file))
    json.dump(ann, open(save_file, 'w'), indent=4)


if __name__ == '__main__':
    make_json()
