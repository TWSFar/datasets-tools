import os
import shutil
import numpy as np
from tqdm import tqdm
import os.path as osp
import matplotlib.pyplot as plt
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring


hyp = {
    'dataset': "dronecc",
    'img_type': '.jpg',
    'mode': 'train',  # for save Set: train.txt
    'data_dir': 'G:\\CV\\Dataset\\Detection\\Visdrone\\VisDrone2020-CC',
    'show': False
}
hyp['txt_dir'] = osp.join(hyp['data_dir'], 'annotations')
hyp['img_dir'] = osp.join(hyp['data_dir'], 'sequences')
# new path
hyp['xml_dir'] = osp.join(hyp['data_dir'], 'Annotations')
hyp['newImg_dir'] = osp.join(hyp['data_dir'], 'JPEGImages')
hyp['set_dir'] = osp.join(hyp['data_dir'], 'ImageSets')


trainSeqDirs = {}
valSeqDirs = {}
testSeqDirs = {}


def show_image(img, labels):
    labels = np.array(labels)
    plt.figure(figsize=(10, 10))
    plt.subplot().imshow(img)
    plt.plot(labels[:, [0, 0, 2, 2, 0]].T, labels[:, [1, 3, 3,  1, 1]].T, '-')
    plt.savefig('uvadt.jpg')
    # plt.show()


def getGTBox(seqdir):
    gts = {}
    with open(osp.join(hyp['txt_dir'], seqdir + '_gt_whole.txt')) as f:
        for line in f.readlines():
            bbox = np.array([int(v.strip()) for v in line.split(',')])
            bbox[4:6] += bbox[2:4]
            frame = str(bbox[0]).zfill(6)
            key = seqdir + '_img' + frame+'.jpg'
            if bbox[8] > 3 or bbox[8] < 1:  # del other classes
                continue
            if key in gts:
                gts[key].append(bbox[[2, 3, 4, 5, 8]])
            else:
                gts[key] = [bbox[[2, 3, 4, 5, 8]]]
    return gts


def make_xml(box_list, image_name, tsize):
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
        node_name.text = str(box_list[i][4])
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
    seqdirs = trainSeqDirs if hyp['mode'] == 'train' else valSeqDirs
    img_list = []
    set_list = []
    for frame_dir in seqdirs:
        img_root = osp.join(hyp['img_dir'], frame_dir)
        for img in os.listdir():
            img_list.append(osp.join(img_root, img))

    for seqdir in tqdm(seqdirs):
        bbox_all = getGTBox(seqdir)
        for img_name, bbox in bbox_all.items():
            if len(bbox) == 0:
                continue
            img_path = osp.join(hyp['img_dir'], img_name.replace('_', '/'))
            tsize = plt.imread(img_path).shape[:2]
            set_list.append(img_name[:-4])

            dom = make_xml(bbox, img_name, tsize)

            # save
            anno_xml = os.path.join(hyp['xml_dir'], img_name[:-4] + '.xml')
            with open(anno_xml, 'w') as fx:
                fx.write(dom.toprettyxml(indent='\t', encoding='utf-8').decode('utf-8'))

            # copy image
            shutil.copy(img_path, hyp['newImg_dir']+'/'+img_name)

            if hyp['show']:
                show_image(plt.imread(img_path), bbox)

    # save ImageSet
    if not osp.exists(hyp['set_dir']):
        os.makedirs(hyp['set_dir'])
    setPath = osp.join(hyp['set_dir'], hyp['mode'] + '.txt')
    with open(setPath, 'w') as f:
        for line in set_list:
            f.writelines(line + '\n')
    print('{} samples generated'.format(len(set_list)))
