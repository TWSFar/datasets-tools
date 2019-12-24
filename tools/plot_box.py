import cv2
import numpy as np


def show_image(img, labels):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1).imshow(img[:, :, ::-1])
    plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [0, 0, 2, 2, 0]].T, '-')
    plt.show()
    pass


img = cv2.imread('G:\\CV\\Code\\tools\\datasets-tools\\VOC2012\\train\\JPEGImages\\00000001.jpg')

bbox = np.array([[100, 100, 700, 1500]])

show_image(img, bbox)


# from matplotlib import pyplot as plot
import cv2
from dataloader import HKBDataset
import numpy as np


imdb = HKBDataset()

img = cv2.imread(imdb.roidb[0]['image'])
bboxes = imdb.roidb[0]['boxes']

for bbox in bboxes:
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    label = 'person|{:.2}'.format(0.25687)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.4 , 1)[0]
    c1 = (x1, y1 - t_size[1]-4)
    c2 = (x1 + t_size[0], y1)
    cv2.rectangle(img, c1, c2, color=#FFFAF0, thickness=-1)
    cv2.putText(img, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    # cv2.putText(img, 'vehicle', (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

cv2.imshow('img', img)
cv2.waitKey(0)