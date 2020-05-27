"""
samples:
    [{'bbox': [[x1, y1, x2, y2], ...],
     'cls': [0, n-1, .....],
     'height': h,
     'width': w,
     'image': <path>},
     ...
     ...
    ]

"""
import os
import cv2
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from sklearn.cluster import KMeans
from datasets.visdrone import VisDrone
from datasets.dotav15 import Dota_V15
# from datasets.visdrone_chip import VisDrone
data_dir = 'G:\\CV\\Dataset\\Detection\\DOTA\\DOTA_V15'
dataset_name = 'dota1.5'
mode = 'train'
result_dir = osp.join("G:\\CV\\Code\\tools\\datasets-tools\\statistics_tools\\result", dataset_name)
if not osp.exists(result_dir):
    os.mkdir(result_dir)
hyp = {
    'small_obj': 32**2 / (640*480),
    'medium_obj': 96**2 / (640*480)
}


def show_image(img, labels=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    # plt.figure(figsize=(10, 10))
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax = plt.Axes(fig, [0, 0, 2, 2])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(img, cmap=cm.jet)
    if labels is not None:
        if labels.shape[0] > 0:
            plt.plot(labels[:, [0, 2, 2, 0, 0]].T, labels[:, [1, 1, 3, 3, 1]].T, '-', color='red')
    plt.savefig("test.png", dpi=600)
    plt.show()
    ax.set_axis_off()


if __name__ == '__main__':
    dataset = Dota_V15(data_dir, mode)
    for sample in dataset.samples:
        img = cv2.imread(sample['image'])
        show_image(img[..., ::-1], sample['bboxes'])