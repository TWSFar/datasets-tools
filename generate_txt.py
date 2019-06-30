import os
import os.path as osp

axis = ['train', 'val']

root = "VisDrone2019-DET-"

txt_path = 'ImagesSet/Main/'

for ax in axis:
    save_path = osp.join(root + ax, txt_path)
    if not osp.exists(save_path):
        os.makedirs(save_path)
    
    save_file = ax + '.txt'

    save = osp.join(save_path, save_file)
    ann = osp.join(root + ax, "annotations")

    file = open(save, 'a')
    annlist = os.listdir(ann)

    for ann in annlist:
        ann = ann.split('.')[0] + '\n'
        file.write(ann)
    file.close()
    
