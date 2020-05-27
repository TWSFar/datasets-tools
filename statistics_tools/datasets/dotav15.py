import os
import cv2
import pickle
import numpy as np
from PIL import Image
import os.path as osp


class Dota_V15(object):
    classes = ('plane', 'ship', 'storage-tank', 'baseball-diamond',
               'tennis-court', 'basketball-court', 'ground-track-field',
               'harbor', 'bridge', 'small-vehicle', 'large-vehicle', 'helicopter',
               'roundabout', 'soccer-ball-field', 'swimming-pool', 'container-crane')
    dataset = 'dota1.5'

    def __init__(self, data_dir, mode):
        self.mode = mode
        # Path
        self.data_dir = osp.join(data_dir, mode)
        self.img_dir = osp.join(self.data_dir, 'images')
        self.ann_dir = osp.join(self.data_dir, 'annotations')
        self.cache_path = self.cre_cache_path(self.data_dir)
        self.cache_file = osp.join(self.cache_path, self.mode + '_samples.pkl')
        self.anno_file = os.path.join(self.ann_dir, '{}.txt')

        # Dataset information
        self.img_type = '.png'
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))
        self.im_ids = self._load_image_set_index()
        self.num_images = len(self.im_ids)

        # bounding boxes and image information
        self.samples = self._load_samples()

    def cre_cache_path(self, data_dir):
        cache_path = osp.join(data_dir, 'cache')
        if not osp.exists(cache_path):
            os.makedirs(cache_path)

        return cache_path

    def _load_image_set_index(self):
        """Load the indexes listed in this dataset's image set file.
        """
        image_index = []
        image_set = os.listdir(self.img_dir)
        for line in image_set:
            image_index.append(line[:-4])  # type of image is .jpg
        return image_index

    def getGTBox(self, anno_path, **kwargs):
        box_all = []
        gt_cls = []
        with open(anno_path, 'r') as f:
            for line in f.readlines():
                data = line.split()
                if (len(data) == 10):
                    box_all.append([float(data[0]), float(data[1]), float(data[4]), float(data[5])])
                    gt_cls.append(self.class_to_id[data[8].strip()])

        return {'bboxes': np.array(box_all, dtype=np.float64),
                'cls': np.array(gt_cls)}

    def _load_samples(self):
        cache_file = self.cache_file
        anno_file = self.anno_file

        # load bbox and save to cache
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                samples = pickle.load(fid)
            print('{} gt samples loaded from {}'.
                  format(self.mode, cache_file))
            return samples

        # load information of image and save to cache
        sizes = [Image.open(osp.join(self.img_dir, index+self.img_type)).size
                 for index in self.im_ids]

        samples = [self.getGTBox(anno_file.format(index))
                   for index in self.im_ids]

        for i, index in enumerate(self.im_ids):
            samples[i]['image'] = osp.join(self.img_dir, index+self.img_type)
            samples[i]['width'] = sizes[i][0]
            samples[i]['height'] = sizes[i][1]

        with open(cache_file, 'wb') as fid:
            pickle.dump(samples, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt samples to {}'.format(cache_file))

        return samples


def show_image(img, bboxes, cls, labels_name):
    for ii, bbox in enumerate(bboxes):
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])

        t_size = cv2.getTextSize(labels[cls[ii]], cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)[0]
        c1 = (x1, y1 - t_size[1]-4)
        c2 = (x1 + t_size[0], y1)
        cv2.rectangle(img, c1, c2, color=(0, 0, 255), thickness=-1)
        cv2.putText(img, labels[cls[ii]], (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
        # cv2.putText(img, 'vehicle', (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    cv2.imshow('img', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    dataset = VisDrone('G:\\CV\\Dataset\\检测\\Visdrone', 'train')
    for i in range(100):
        sample = dataset.samples[i]
        img_path = sample['image']
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        bboxes = sample['bboxes']
        cls = sample['cls']
        labels = dataset.classes
        show_image(img, bboxes, cls, labels)
        # cv2.imread(sample[''])
        pass
