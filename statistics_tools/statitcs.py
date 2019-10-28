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
import numpy as np
import os.path as osp
from datasets.visdrone import VisDrone
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from sklearn.cluster import KMeans
result = "G:\\CV\\Code\\tools\\datasets-tools\\statistics_tools\\result"
data_dir = 'G:\\CV\\Dataset\\检测\\Visdrone'
dataset_name = 'visdrone'
mode = 'train'

hyp = {
    'small_obj': 32**2 / (640*480),
    'medium_obj': 96**2 / (640*480)
}


def analyse_labels(dataset):
    """
    mainly analyse labels
    args:
        per_img_obj:
            every image contain object's number
        obj_img_num:
            idx object contatin [idx] image
        per_cls_num:
            evary classes contain object
    """
    labels = dataset.samples
    num_cls = dataset.num_classes
    labels_info = osp.join(result, dataset_name, mode + '_labels_info.txt')
    f = open(labels_info, 'w')

    """ image's number for per number of obj """
    per_img_obj = [len(label['cls']) for label in labels]
    per_img_obj = np.array(per_img_obj, dtype=np.int64)
    obj_img_num = np.bincount(per_img_obj)
    x = np.where(obj_img_num > 0)[0]
    y = obj_img_num[x]

    # plot
    plt.figure(figsize=(14, 8))
    plt.grid()
    plt.title('There are Yi images containing Xi targets')
    plt.xlabel('number of object')
    plt.ylabel('number of image')
    my_x_ticks = np.arange(0, len(obj_img_num), 30)
    plt.xticks(my_x_ticks)
    plt.bar(x, y)
    plt.savefig(osp.join(result, dataset_name, mode + "_obj_img_num.png"))
    plt.close()

    """ number of every class """
    per_img_obj = np.sort(per_img_obj)[::-1]
    ymax = per_img_obj.max()
    ymean = per_img_obj.mean()
    ymedian = np.median(per_img_obj)
    ymode = stats.mode(per_img_obj)[0][0]
    pprint = 'max: {}\nmean: {:.1}\nmedian: {}\nmode: {}'.format(ymax, ymean, ymedian, ymode)

    # plot
    plt.figure(figsize=(14, 8))
    plt.grid()
    plt.title('There are Xi\'th image containing Yi targets')
    plt.xlabel('image')
    plt.ylabel('number of object')
    my_x_ticks = np.arange(0, len(per_img_obj), 300)
    my_y_ticks = np.arange(0, ymax+5, 20)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.bar(range(len(per_img_obj)), per_img_obj)
    plt.text(900, 300, pprint)
    plt.savefig(osp.join(result, dataset_name, mode + "_per_img_obj.png"))
    plt.close()

    """" number of per classes """
    classes = []
    for label in labels:
        classes.extend(label['cls'])
    per_cls_num = np.bincount(np.array(classes, dtype=np.int64), minlength=num_cls)

    # write
    for ii, label_name in enumerate(dataset.classes):
        temp = label_name + ': {}'.format(per_cls_num[ii])
        f.writelines(temp+'\n')
    f.writelines('-------------\n')
    f.writelines('sum obj: {}\n'.format(per_cls_num.sum()))
    f.writelines('mean per classes: {}\n'.format(per_cls_num.mean()))
    f.writelines('mean obj per image: {}\n'.format(ymean))
    f.writelines('median obj per image: {}\n'.format(ymedian))
    f.writelines('max obj per image: {}\n'.format(ymax))
    f.writelines('mode obj per image: {}\n'.format(ymode))
    f.writelines('-------------\n')

    # plot
    plt.figure(figsize=(14, 8))
    plt.grid()
    plt.title('number of evry classes')
    plt.xlabel('class id')
    plt.ylabel('number of class')
    my_x_ticks = np.arange(0, num_cls, 1)
    plt.xticks(my_x_ticks)
    plt.bar(range(num_cls), per_cls_num)
    plt.savefig(osp.join(result, dataset_name, mode + "_per_cls_num.png"))
    plt.close()

    if dataset_name == 'visdrone':
        temp = []
        for label in labels:
            temp.extend(label['ignore_cls'])
        temp = np.array(temp, dtype=np.int64)
        f.writelines("visdrone ignore_region: {}\n".format(sum(temp == 0)))
        f.writelines("visdrone others: {}\n".format(sum(temp == 11)))

    f.close()
    pass


def analyse_bboxes(dataset):
    labels_info = osp.join(result, dataset_name, mode + '_bbox_info.txt')
    f = open(labels_info, 'w')

    samples = dataset.samples
    # object size
    obj_size_rw = []
    obj_size_rh = []
    per_obj_size_rw = [[] for i in range(dataset.num_classes)]
    per_obj_size_rh = [[] for i in range(dataset.num_classes)]
    per_img_obj_side_max_diff = []
    per_img_obj_area_max_diff = []

    for jj, sample in enumerate(tqdm(samples)):
        img_w = sample['width']
        img_h = sample['height']
        max_side = 0
        min_side = 1
        max_area = 0
        min_area = 1
        for ii, bbox in enumerate(sample['bboxes']):
            if isinstance(bbox, np.ndarray):
                bbox = bbox.tolist()
            cls_id = sample['cls'][ii]
            # 640, if area < 32 * 32 , obj is small obj in coco
            rw = (bbox[2] - bbox[0]) / img_w
            rh = (bbox[3] - bbox[1]) / img_h
            max_side = max(rw, rh, max_side)
            min_side = min(rw, rh, min_side)
            max_area = max(rw * rh, max_area)
            min_area = min(rw * rh, min_area)
            if rw <= 0 or rh <= 0:
                continue
            obj_size_rw.append(rw)
            obj_size_rh.append(rh)
            per_obj_size_rw[cls_id].append(rw)
            per_obj_size_rh[cls_id].append(rh)
        per_img_obj_side_max_diff.append(max_side - min_side)
        per_img_obj_area_max_diff.append(max_area - min_area)

    """ all obj and use ratio """
    per_img_obj_side_max_diff = np.array(per_img_obj_side_max_diff)
    per_img_obj_area_max_diff = np.array(per_img_obj_area_max_diff)
    obj_size_rw = np.array(obj_size_rw, dtype=np.float64)
    obj_size_rh = np.array(obj_size_rh, dtype=np.float64)
    obj_area_ratio = obj_size_rh * obj_size_rw
    obj_longest_side = np.maximum(obj_size_rh, obj_size_rw)

    # kmean get ratio and size
    obj_max_wh_ratio = (obj_size_rw / obj_size_rh).reshape(-1, 1)
    estimator = KMeans(n_clusters=3)
    estimator.fit(obj_max_wh_ratio)
    k_center = estimator.cluster_centers_.reshape(-1)
    f.writelines('kmeans ratio: {}, {}, {}\n'.format(*k_center))

    estimator.fit(obj_area_ratio.reshape(-1, 1))
    k_center = estimator.cluster_centers_.reshape(-1)
    f.writelines('kmeans scale: {}, {}, {}\n'.format(*k_center))

    # write
    f.writelines('\nall object bboxes information:\n')
    f.writelines('------area-------\n')
    area_max = obj_area_ratio.max()
    area_mean = obj_area_ratio.mean()
    area_min = obj_area_ratio.min()
    area_diff_max = per_img_obj_area_max_diff.max()
    area_diff_mean = per_img_obj_area_max_diff.mean()
    area_diff_min = per_img_obj_area_max_diff.min()
    f.writelines('maximum object area: {}\n'.format(area_max))
    f.writelines('mean object area: {}\n'.format(area_mean))
    f.writelines('minimum object area: {}\n'.format(area_min))
    f.writelines('maximum object area diff: {}\n'.format(area_diff_max))
    f.writelines('mean object area diff: {}\n'.format(area_diff_mean))
    f.writelines('minimum object area diff: {}\n'.format(area_diff_min))
    f.writelines('-------side------\n')
    side_max = obj_longest_side.max()
    side_mean = obj_longest_side.mean()
    side_min = obj_longest_side.min()
    side_diff_max = per_img_obj_side_max_diff.max()
    side_diff_mean = per_img_obj_side_max_diff.mean()
    side_diff_min = per_img_obj_side_max_diff.min()
    f.writelines('maximum object side: {}\n'.format(side_max))
    f.writelines('mean object side: {}\n'.format(side_mean))
    f.writelines('minimum object side: {}\n'.format(side_min))
    f.writelines('maximum object side diff: {}\n'.format(side_diff_max))
    f.writelines('mean object side diff: {}\n'.format(side_diff_mean))
    f.writelines('minimum object side diff: {}\n'.format(side_diff_min))
    f.writelines('-------scale: s, m, l------\n')
    small_obj = (obj_area_ratio < hyp['small_obj']).sum() / len(obj_area_ratio)
    large_obj = (obj_area_ratio > hyp['medium_obj']).sum() / len(obj_area_ratio)
    medium_obj = 1 - small_obj - large_obj
    f.writelines('small object percentage: {:.4}%\n'.format(small_obj * 100))
    f.writelines('medium object percentage: {:.4}%\n'.format(medium_obj * 100))
    f.writelines('large object percentage: {:.4}%\n'.format(large_obj * 100))
    f.writelines('------------------\n\n')

    # # plot w and h
    # plt.figure()
    # plt.title('obj size (ratio)')
    # plt.grid()
    # plt.xlabel('width')
    # plt.ylabel('hight')
    # my_x_ticks = np.arange(0, 1, 0.1)
    # my_y_ticks = np.arange(0, 1, 0.1)
    # plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)
    # plt.bar(obj_size_rw, obj_size_rh)
    # plt.savefig(osp.join(result, dataset_name, mode+"_obj_size_ratio.png"))
    # # plt.show()
    # plt.close()

    # plot area
    obj_area_ratio_sample = np.sort(obj_area_ratio)[::-1]
    obj_area_ratio_sample = obj_area_ratio_sample[::100]
    plt.figure()
    plt.title('obj area (ratio sample 1/100)')
    plt.grid()
    plt.xlabel('image')
    plt.ylabel('area')
    plt.scatter(range(len(obj_area_ratio_sample)), obj_area_ratio_sample)
    plt.savefig(osp.join(result, dataset_name, mode+"_obj_area_ratio.png"))
    # plt.show()
    plt.close()

    """ obj for per cls (use ratio)"""
    cls_obj_mean_area = []
    for cls_id in range(dataset.num_classes):
        cls_obj_rw = np.array(per_obj_size_rw[cls_id], dtype=np.float64)
        cls_obj_rh = np.array(per_obj_size_rh[cls_id], dtype=np.float64)
        cls_obj_area = cls_obj_rh * cls_obj_rw
        cls_obj_longest_side = np.maximum(cls_obj_rh, cls_obj_rw)

        # write
        f.writelines('{}: {}\n'.format(cls_id, dataset.classes[cls_id]))
        area_max = cls_obj_area.max() if len(cls_obj_area) > 0 else 0
        area_mean = cls_obj_area.mean() if len(cls_obj_area) > 0 else 0
        area_min = cls_obj_area.min() if len(cls_obj_area) > 0 else 0
        f.writelines('\tarea:\n')
        f.writelines('\t\tmaximum object area: {}\n'.format(area_max))
        f.writelines('\t\tmean object area: {}\n'.format(area_mean))
        f.writelines('\t\tminimum object area: {}\n'.format(area_min))
        side_max = cls_obj_longest_side.max() if len(cls_obj_longest_side) > 0 else 0
        side_mean = cls_obj_longest_side.mean() if len(cls_obj_longest_side) > 0 else 0
        side_min = cls_obj_longest_side.min() if len(cls_obj_longest_side) > 0 else 0
        f.writelines('\tside:\n')
        f.writelines('\t\tmaximum object side: {}\n'.format(side_max))
        f.writelines('\t\tmean object side: {}\n'.format(side_mean))
        f.writelines('\t\tminimum object side: {}\n'.format(side_min))
        f.writelines('------------------\n\n')

        cls_obj_mean_area.append(area_mean)

    """ plot every classes object mean area """
    plt.figure()
    plt.title('cls mean area (ratio)')
    plt.grid()
    plt.xlabel('classes id')
    plt.ylabel('mean area')
    my_x_ticks = np.arange(0, dataset.num_classes, 1)
    plt.xticks(my_x_ticks)
    plt.bar(range(dataset.num_classes), cls_obj_mean_area)
    plt.savefig(osp.join(result, dataset_name, mode+"_cls_mean_area.png"))
    # plt.show()
    plt.close()


def analyse_img_MeanAndStd(dataset):
    meanAndStd_info = osp.join(result, 'train_dataset_mean_std.txt')
    samples = dataset.samples
    Rmean = []
    Gmean = []
    Bmean = []

    Rstd = []
    Gstd = []
    Bstd = []

    for sample in tqdm(samples):
        img_path = sample['image']
        img = plt.imread(img_path)

        Rmean.append(img[..., 0].mean())
        Gmean.append(img[..., 1].mean())
        Bmean.append(img[..., 2].mean())

        Rstd.append(np.std(img[..., 0]))
        Gstd.append(np.std(img[..., 1]))
        Bstd.append(np.std(img[..., 2]))

    with open(meanAndStd_info, 'w') as f:
        f.writelines('R mean: {}\n'.format(np.array(Rmean).mean()))
        f.writelines('G mean: {}\n'.format(np.array(Gmean).mean()))
        f.writelines('B mean: {}\n'.format(np.array(Bmean).mean()))
        f.writelines('R std: {}\n'.format(np.array(Rstd).mean()))
        f.writelines('G std: {}\n'.format(np.array(Gstd).mean()))
        f.writelines('B std: {}\n'.format(np.array(Bstd).mean()))


def statics(dataset):
    # analyse_labels(dataset)
    analyse_bboxes(dataset)
    # if mode == 'train':
        # analyse_img_MeanAndStd(dataset)  # olny analys


if __name__ == '__main__':
    dataset = VisDrone(data_dir, mode)
    statics(dataset)