import numpy as np
box_colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
              (0, 255, 255), (255, 255, 255),
              (125, 125, 125), (0, 0, 0),
              (0, 125, 255), (0.439, 0.502, 0.412),
              (0, 0, 0)  # others
              )

def getGTBox_VisDrone(anno_path, **kwargs):
    box_all = []
    gt_cls = []
    with open(anno_path, 'r') as f:
        data = [x.strip().split(',')[:8] for x in f.readlines()]
        annos = np.array(data)

    bboxes = annos[annos[:, 4] == '1'][:, :6].astype(np.float64)
    for bbox in bboxes:
        if bbox[2] <= 0 or bbox[3] <= 0:
            print(osp.basename(anno_path) + ' exist an illegal side:')
            print(bbox)
            print('illegal side has been abandoned')
            continue
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        box_all.append(bbox[:4].tolist())
        gt_cls.append(int(bbox[5]) - 1)  # index begin idx 0

    return box_all, gt_cls

def show_image(img, labels=None):
    import matplotlib.pyplot as plt
    # labels = np.array(labels)
    plt.figure(figsize=(10, 10))
    plt.imshow(img[..., ::-1])
    # plt.plot(labels[:, [0, 0, 2, 2, 0]].T, labels[:, [1, 3, 3,  1, 1]].T, '-')
    # plt.savefig('mask.jpg')
    plt.show()

def plot_img(img, bboxes, clss):
    # img = img.astype(np.float64) / 255.0 if img.max() > 1.0 else img
    for i, bbox in enumerate(bboxes):
        try:
            if -1 in bbox:
                continue
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            id = clss[i]
            # score = float(bbox[5])
            # if score <= 0.3:
            #     continue
            label = str(id)
            # plot
            box_color = box_colors[min(id-1, len(box_colors)-1)]
            text_color = (1, 1, 1)
            cv2.putText(img, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX, 0.5, text_color, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=box_color, thickness=1)

        except Exception as e:
            print(e)
            continue

    return img
# 文件名: 0000074_05715_d_0000011.jpg [51/1610]
# 图片大小: 265.7KB
# 修改日期: 2018/03/10 21:39:27
# 图片信息: 1920x1080 (Jpeg,YUV)

# 文件名: 9999938_00000_d_0000051.jpg [303/1610]
# 图片大小: 219.8KB
# 修改日期: 2018/03/10 21:39:27
# 图片信息: 1400x788 (Jpeg,YUV)



import cv2
import os
temp = "G:\\CV\\Dataset\\Detection\\Visdrone\\test\images\\"
img_dir = "G:\\CV\\Dataset\\Detection\\Visdrone\\test\images\\{}.jpg"
anno_dir = "G:\\CV\\Dataset\\Detection\\Visdrone\\test\\annotations\\{}.txt"
img_list = os.listdir(temp)
for ii, img_name in enumerate(img_list):
    if ii % 10 != 0:
        continue
    img_path = img_dir.format(img_name[:-4])
    anno_path = anno_dir.format(img_name[:-4])
    img = cv2.imread(img_path)
    bbox, clss = getGTBox_VisDrone(anno_path)
    img = plot_img(img, bbox, clss)
    show_image(img)