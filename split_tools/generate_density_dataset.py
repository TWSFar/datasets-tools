"""convert VOC format
+ density_voc
    + JPEGImages
    + SegmentationClass
"""

import os, sys
import glob
import cv2
import random
import shutil
import argparse
import numpy as np
import pandas as pd
import os.path as osp
import concurrent.futures
from tqdm import tqdm

from datasets import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="convert to voc dataset")
    parser.add_argument('--dataset', type=str, default='VisDrone',
                        choices=['VisDrone'], help='dataset name')
    parser.add_argument('--mode', type=list, default=['val'],
                        choices=['train', 'val', 'test'], help='for train or test')
    parser.add_argument('--db_root', type=str, default="E:\\CV\\data\\visdrone",
                        help="dataset's root path")
    parser.add_argument('--mask_size', type=list, default=[30, 40],
                        help="Size of production target mask")
    parser.add_argument('--show', type=bool, default=False,
                        help="show image and region mask")
    args = parser.parse_args()
    return args


def show_image(img, labels, mask):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1).imshow(img)
    plt.plot(labels[:, [0, 0, 2, 2, 0]].T, labels[:, [1, 3, 3,  1, 1]].T, '-')
    plt.subplot(2, 1, 2).imshow(mask)
    # plt.savefig('test_0.jpg')
    plt.show()


# copy train and test images
def _copy(src_image, dest_path):
    shutil.copy(src_image, dest_path)


def _myaround_up(value):
    """0.2 * stride = 3.2"""
    tmp = np.floor(value).astype(np.int32)
    return tmp + 1 if value - tmp > 0.2 else tmp


def _myaround_down(value):
    """0.2 * stride = 3.2"""
    tmp = np.ceil(value).astype(np.int32)
    return max(0, tmp - 1 if tmp - value > 0.2 else tmp)


def _generate_mask(sample, mask_scale=(30, 40)):
    try:
        height, width = sample["height"], sample["width"]

        # Chip mask 40 * 30, model input size 640x480
        mask_h, mask_w = mask_scale
        density_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)

        for box in sample["bboxes"]:
            xmin = _myaround_down(1.0 * box[0] / width * mask_w)
            ymin = _myaround_down(1.0 * box[1] / height * mask_h)
            xmax = _myaround_up(1.0 * box[2] / width * mask_w)
            ymax = _myaround_up(1.0 * box[3] / height * mask_h)
            density_mask[ymin:ymax, xmin:xmax] += 1

        return density_mask

    except Exception as e:
        print(e)
        print(sample["images"])


if __name__ == "__main__":
    args = parse_args()

    dataset = get_dataset(args.dataset, args.db_root)
    dest_datadir = args.db_root + "/density_voc"
    image_dir = dest_datadir + '/JPEGImages'
    mask_dir = dest_datadir + '/SegmentationClass'
    annotation_dir = dest_datadir + '/Annotations'
    list_folder = dest_datadir + '/ImageSets'

    if not os.path.exists(dest_datadir):
        os.mkdir(dest_datadir)
        os.mkdir(image_dir)
        os.mkdir(mask_dir)
        os.mkdir(annotation_dir)
        os.mkdir(list_folder)

    for split in args.mode:
        img_list = dataset._get_imglist(split)
        samples = dataset._load_samples(split)

        with open(os.path.join(list_folder, split + '.txt'), 'w') as f:
            temp = [os.path.basename(x)[:-4]+'\n' for x in img_list]
            f.writelines(temp)

        print('copy {} images....'.format(split))
        with concurrent.futures.ThreadPoolExecutor() as exector:
            exector.map(_copy, img_list, [image_dir]*len(img_list))

        if split == "train" or split == 'val':
            print('generate {} masks...'.format(split))
            for sample in tqdm(samples):
                density_mask = _generate_mask(sample, args.mask_size)
                maskname = osp.join(mask_dir, osp.basename(sample['image']).
                                replace('jpg', 'png'))
                cv2.imwrite(maskname, density_mask)

                if args.show:
                    img = cv2.imread(sample['image'])
                    show_image(img, sample['bboxes'], density_mask)

            print('copy {} box annos...'.format(split))
            anno_list = dataset._get_annolist(split)
            with concurrent.futures.ThreadPoolExecutor() as exector:
                exector.map(_copy, anno_list, [annotation_dir]*len(anno_list))

        print('done.')
