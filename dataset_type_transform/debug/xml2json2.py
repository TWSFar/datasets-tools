"""Convert PASCAL VOC annotations to MSCOCO format and save to a json file.
The MSCOCO annotation has following structure:
{
    "images": [
        {
            "file_name": ,
            "height": ,
            "width": ,
            "id":
        },
        ...
    ],
    "type": "instances",
    "annotations": [
        {
            "segmentation": [],
            "area": ,
            "iscrowd": ,
            "image_id": ,
            "bbox": [],
            "category_id": ,
            "id": ,
            "ignore":
        },
        ...
    ],
    "categories": [
        {
            "supercategory": ,
            "id": ,
            "name":
        },
        ...
    ]
}
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import os.path as osp
from collections import OrderedDict
import json
import shutil
import xmltodict
import mmcv
import concurrent.futures

from datasets import VisDrone

logger = logging.getLogger(__name__)


def _copy(src_image, dest_path):
    shutil.copy(src_image, dest_path)


class PASCALVOC2COCO(object):
    """Converters that convert PASCAL VOC annotations to MSCOCO format."""

    def __init__(self):
        self.cat2id = {
            '1': 0, '2': 1, '3': 2, '4': 3, '5': 4,
            '6': 5, '7': 6, '8': 7, '9': 8,
            '10': 9,
        }

    def get_img_item(self, file_name, image_id, size):
        """Gets a image item."""
        image = OrderedDict()
        image['file_name'] = file_name
        image['height'] = int(size['height'])
        image['width'] = int(size['width'])
        image['id'] = image_id
        return image

    def get_ann_item(self, obj, image_id, ann_id):
        """Gets an annotation item."""
        x1 = int(obj['bndbox']['xmin']) - 1
        y1 = int(obj['bndbox']['ymin']) - 1
        w = int(obj['bndbox']['xmax']) - x1
        h = int(obj['bndbox']['ymax']) - y1

        annotation = OrderedDict()
        annotation['segmentation'] = [[x1, y1, x1, (y1 + h), (x1 + w), (y1 + h), (x1 + w), y1]]
        annotation['area'] = w * h
        annotation['iscrowd'] = 0
        annotation['image_id'] = image_id
        annotation['bbox'] = [x1, y1, w, h]
        annotation['category_id'] = self.cat2id[obj['name']]
        annotation['id'] = ann_id
        annotation['ignore'] = int(obj['difficult'])
        return annotation

    def get_cat_item(self, name, id):
        """Gets an category item."""
        category = OrderedDict()
        category['supercategory'] = 'none'
        category['id'] = id
        category['name'] = name
        return category

    def convert(self, devkit_path, split):
        """Converts PASCAL VOC annotations to MSCOCO format. """
        split_file = osp.join(devkit_path, 'ImageSets/Main/{}.txt'.format(split))
        ann_dir = osp.join(devkit_path, 'Annotations')

        name_list = mmcv.list_from_file(split_file)

        # copy image
        split_imgdir = os.path.join(coco_imgdir, split)
        if os.path.exists(split_imgdir):
            shutil.rmtree(split_imgdir)
        os.mkdir(split_imgdir)
        img_list = [os.path.join(image_dir, name+'.jpg') for name in name_list]
        with concurrent.futures.ThreadPoolExecutor() as exector:
            exector.map(_copy, img_list, [split_imgdir]*len(img_list))

        images, annotations = [], []
        ann_id = 1
        for id, name in enumerate(name_list):
            image_id = id

            xml_file = osp.join(ann_dir, name + '.xml')

            with open(xml_file, 'r') as f:
                ann_dict = xmltodict.parse(f.read(), force_list=('object',))

            if 'object' in ann_dict['annotation']:
                # Add image item.
                image = self.get_img_item(name + '.jpg', image_id, ann_dict['annotation']['size'])
                images.append(image)

                for obj in ann_dict['annotation']['object']:
                    # Add annotation item.
                    annotation = self.get_ann_item(obj, image_id, ann_id)
                    annotations.append(annotation)
                    ann_id += 1
            else:
                logger.warning('{} does not have any object'.format(name))

        categories = []
        for name, id in self.cat2id.items():
            # Add category item.
            category = self.get_cat_item(name, id)
            categories.append(category)

        ann = OrderedDict()
        ann['images'] = images
        ann['type'] = 'instances'
        ann['annotations'] = annotations
        ann['categories'] = categories

        save_file = os.path.join(coco_dir, 'annotations/instances_{}.json'.format(split))
        logger.info('Saving annotations to {}'.format(save_file))
        with open(save_file, 'w') as f:
            json.dump(ann, f)


if __name__ == '__main__':
    dataset = VisDrone()
    voc_dir = dataset.detect_voc_dir
    image_dir = voc_dir + '/JPEGImages'
    list_dir = voc_dir + '/ImageSets/Main'
    anno_dir = voc_dir + '/Annotations'

    coco_dir = dataset.detect_coco_dir
    coco_imgdir = coco_dir + '/images'
    coco_annodir = coco_dir + '/annotations'

    if not os.path.exists(coco_dir):
        os.mkdir(coco_dir)
        os.mkdir(coco_imgdir)
        os.mkdir(coco_annodir)

    converter = PASCALVOC2COCO()
    devkit_path = voc_dir

    for split in ['val']:
        converter.convert(devkit_path, split)
