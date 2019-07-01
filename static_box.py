import xml.etree.ElementTree as ET
import os
import os.path as osp
import sklearn as sk
from tqdm import tqdm

if __name__ == "__main__":
    # all annotations of datasets
    path = "unchange/Annotations"
    
    # create result file
    res = "result"
    if not osp.exists(res):
        os.makedirs(res)

    box_number = 0
    means_w = 0
    means_h = 0
    filelist = os.listdir(path)

    w_h_minset = []
    w_div_h = []
    boxes_wh = []
    res_file = ["w_h_minset.txt", "w_div_h.txt", "boxes_wh.txt"]

    for file in tqdm(filelist):
        filepath = osp.join(path, file)
        tree = ET.parse(filepath)
        size=tree.find('size')

        width = float(size.find('width').text)
        height = float(size.find('height').text)

        objs = tree.findall('object')

        #box_wh = []
        

        for ix, obj in enumerate(objs):
            cls = obj.find('name').text
            
            #box.append(float(cls))

            bbox = obj.find('bndbox')

            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)

            bw = (x2 - x1) / width
            bh = (y2 - y1) / height

            w_div_h.append([bw / bh])
            w_h_minset.append([min(bw, bh)])
            #box_wh.append([bw, bh, width, height])
            boxes_wh.append([bw, bh, width, height])

            box_number += 1
            temp = (box_number-1) / box_number
            means_h = (means_h * temp + height/box_number) 
            means_w = (means_w * temp + width/box_number) 
            
    result = (w_h_minset, w_div_h, boxes_wh)

    print(means_h, means_w)

    for i, txt in enumerate(res_file):
        temp = osp.join(res, txt)
        file = open(temp, 'a')
        data = result[i]
        for line in data:
            for j in range(len(line)):
                if j!= 0: file.write(' ')
                file.write(str(line[j]))
            file.write('\n')
        file.close()
