import os
import cv2
import json
import pickle
import numpy as np
from PIL import Image
import os.path as osp


class TT100K(object):
    classes = ('i1,i10,i11,i12,i13,i14,i15,i2,i3,i4,i5,il100,il110,il50,'
              'il60,il70,il80,il90,io,ip,p1,p10,p11,p12,p13,p14,p15,p16,'
              'p17,p18,p19,p2,p20,p21,p22,p23,p24,p25,p26,p27,p28,p3,p4,'
              'p5,p6,p7,p8,p9,pa10,pa12,pa13,pa14,pa8,pb,pc,pg,ph1.5,ph2,'
              'ph2.1,ph2.2,ph2.4,ph2.5,ph2.8,ph2.9,ph3,ph3.2,ph3.5,ph3.8,'
              'ph4,ph4.2,ph4.3,ph4.5,ph4.8,ph5,ph5.3,ph5.5,pl10,pl100,pl110,'
              'pl120,pl15,pl20,pl25,pl30,pl35,pl40,pl5,pl50,pl60,pl65,pl70,'
              'pl80,pl90,pm10,pm13,pm15,pm1.5,pm2,pm20,pm25,pm30,pm35,pm40,'
              'pm46,pm5,pm50,pm55,pm8,pn,pne,po,pr10,pr100,pr20,pr30,pr40,'
              'pr45,pr50,pr60,pr70,pr80,ps,pw2,pw2.5,pw3,pw3.2,pw3.5,pw4,pw4.2,'
              'pw4.5,w1,w10,w12,w13,w16,w18,w20,w21,w22,w24,w28,w3,w30,w31,w32,'
              'w34,w35,w37,w38,w41,w42,w43,w44,w45,w46,w47,w48,w49,w5,w50,w55,'
              'w56,w57,w58,w59,w60,w62,w63,w66,w8,wo,i6,i7,i8,i9,ilx,p29,w29,'
              'w33,w36,w39,w4,w40,w51,w52,w53,w54,w6,w61,w64,w65,w67,w7,w9,'
              'pax,pd,pe,phx,plx,pmx,pnl,prx,pwx,w11,w14,w15,w17,w19,w2,w23,'
              'w25,w26,w27,pl0,pl4,pl3,pm2.5,ph4.4,pn40,ph3.3,ph2.6').split(',')
    dataset = 'tt100k'

    def __init__(self, data_dir, mode):
        self.mode = mode

        # File
        if self.mode == 'train':
            root_dir = 'train'
        elif self.mode in ['val', 'test']:
            root_dir = 'test'

        # Path
        self.data_dir = data_dir
        self.img_dir = osp.join(self.data_dir, root_dir)
        self.cache_path = self.cre_cache_path(self.data_dir)
        self.cache_file = osp.join(self.cache_path, self.mode + '_samples.pkl')
        self.anno_file = os.path.join(self.data_dir, 'annotations.json')

        # Dataset information
        self.img_type = '.jpg'
        self.img_ids = self._load_image_set_index()
        self.num_images = len(self.img_ids)
        with open(self.anno_file, 'r') as f:
            labels = json.load(f)
        self.classes = labels['types']
        self.anno = labels['imgs']
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))

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

    def getGTBox(self, index):
        objects = self.anno[str(index)]['objects']
        box_all = []
        gt_cls = []
        temp = ['xmin', 'ymin', 'xmax', 'ymax']
        for obj in objects:
            box = []
            for lab in temp:
                box.append(obj['bbox'][lab])
            box_all.append(box)
            gt_cls.append(self.class_to_id[obj['category']])

        return {'bboxes': np.array(box_all, dtype=np.float64),
                'cls': np.array(gt_cls)}

    def _load_samples(self):
        cache_file = self.cache_file

        # load bbox and save to cache
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                samples = pickle.load(fid)
            print('{} gt samples loaded from {}'.
                  format(self.mode, cache_file))
            return samples

        # load information of image and save to cache
        sizes = [Image.open(osp.join(self.img_dir, index+self.img_type)).size
                 for index in self.img_ids]

        samples = [self.getGTBox(index) for index in self.img_ids]

        for i, index in enumerate(self.img_ids):
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

    cv2.namedWindow("enhanced", 0)
    cv2.resizeWindow("enhanced", 640, 480)
    cv2.imshow("enhanced", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    dataset = TT100K('G:\\CV\\Dataset\\Detection\\TT100K\\', 'train')
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
