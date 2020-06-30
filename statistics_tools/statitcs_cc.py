import os
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets.dronecc import DroneCC
mode = 'train'
dataset_name = "dronecc"
data_root = 'G:\\CV\\Dataset\\CC\\Visdrone\\VisDrone2020-CC'
result_dir = osp.join("G:\\CV\\Code\\tools\\datasets-tools\\statistics_tools\\result", dataset_name)
if not osp.exists(result_dir):
    os.mkdir(result_dir)


def statistic_nums(samples):
    nums = []
    for sample in tqdm(samples):
        nums.append(len(sample["coordinate"]))
    nums = np.array(nums)
    with open(osp.join(result_dir, 'gt_info.txt'), 'w') as f:
        f.writelines('sum obj: {}\n'.format(nums.sum()))
        f.writelines('mean obj per image: {}\n'.format(nums.mean()))
        f.writelines('median obj per image: {}\n'.format(np.median(nums)))
        f.writelines('max obj per image: {}\n'.format(nums.max()))
        f.writelines('mode obj per image: {}\n'.format(nums.min()))


def analyse_img_MeanAndStd(samples):
    meanAndStd_info = osp.join(result_dir, 'RGB.txt')
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


def main():
    dataset = DroneCC(data_root, mode)
    samples = dataset.samples
    statistic_nums(samples)
    analyse_img_MeanAndStd(samples)

if __name__ == '__main__':
    main()
    