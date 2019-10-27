"""
visdrone:
    if we use mmdetection train and val,
    id begin idx 1 is best. becase, in mmdetection, the default
    classes's number is n+1, we don't know the background's idx.
    However, in other detector, the idx begin 0 or 1 is need conside.

    note:
        classes = (
            '_background_0', 'pedestrian', 'person', 'bicycle', 'car',
            'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
"""
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

hyp = {
    'dataset': 'VisDrone2019',
    'img_type': '.jpg',
    'mode': 'val',  # for save Set: train.txt
    'data_dir': 'G:\\CV\\Dataset\\检测\\Visdrone\\VisDrone2019-DET-val',
}
hyp['xml_dir'] = osp.join(hyp['data_dir'], 'annotations_xml')
hyp['txt_dir'] = osp.join(hyp['data_dir'], 'annotations')
hyp['img_dir'] = osp.join(hyp['data_dir'], 'images')
hyp['set_dir'] = osp.join(hyp['data_dir'], 'ImageSets', 'Main')


def getGTBox_VisDrone(anno_path, **kwargs):
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


def make_xml(box_list, label_list, image_name, tsize):
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = hyp['dataset']

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_object_num = SubElement(node_root, 'object_num')
    node_object_num.text = str(len(box_list))

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(tsize[1])  # tsize: (h, w)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(tsize[0])  # tsize: (h, w)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for i in range(len(box_list)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = str(label_list[i])
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        # voc dataset is 1-based
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(box_list[i][0]) + 1)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(box_list[i][1]) + 1)
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(box_list[i][2] + 1))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(box_list[i][3] + 1))

    xml = tostring(node_root, encoding='utf-8')
    dom = parseString(xml)
    # print(xml)
    return dom


if __name__ == '__main__':
    setList = []
    if not osp.exists(hyp['xml_dir']):
        os.makedirs(hyp['xml_dir'])

    txt_files = os.listdir(hyp['txt_dir'])

    for file_name in tqdm(txt_files):
        setList.append(file_name[:-4])

        anno_txt = os.path.join(hyp['txt_dir'], file_name)
        box_all, gt_cls = getGTBox_VisDrone(anno_txt)

        # image info
        img_name = file_name[:-4] + hyp['img_type']  # image name
        img_path = osp.join(hyp['img_dir'], img_name)  # image path
        tsize = plt.imread(img_path).shape[:2]

        dom = make_xml(box_all, gt_cls, img_name, tsize)

        # save
        anno_xml = os.path.join(hyp['xml_dir'], file_name[:-4] + '.xml')
        with open(anno_xml, 'w') as fx:
            fx.write(dom.toprettyxml(indent='\t', encoding='utf-8').decode('utf-8'))

    # save ImageSet
    if not osp.exists(hyp['set_dir']):
        os.makedirs(hyp['set_dir'])
    setPath = osp.join(hyp['set_dir'], hyp['mode'] + '.txt')
    with open(setPath, 'w') as f:
        for line in setList:
            f.writelines(line + '\n')
    pass
