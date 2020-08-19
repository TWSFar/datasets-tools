import os
import cv2
import os.path as osp
import numpy as np
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString


def make_xml(box_list, image_name, tsize):
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = "VisDrone2019"

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_object_num = SubElement(node_root, 'object_num')
    node_object_num.text = str(len(box_list))

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(tsize[1])  # tsize: (h, w)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(tsize[0])  # tsize: (h, w)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for i in range(len(box_list)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = str(box_list[i][4])
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        # voc dataset is 1-based
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(box_list[i][0]) + 1)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(box_list[i][1]) + 1)
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(box_list[i][2] + 1))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(box_list[i][3] + 1))

    xml = tostring(node_root, encoding='utf-8')
    dom = parseString(xml)
    # print(xml)
    return dom


def show_image(img, labels):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(img[..., ::-1])
    plt.plot(labels[:, [0, 0, 2, 2, 0]].T, labels[:, [1, 3, 3,  1, 1]].T, '-')
    # plt.savefig('mask.jpg')
    plt.show()


pool_dir = "G:\\CV\\Dataset\\Detection\\Visdrone\\TEMP\\ship_trash"
ship_dir = "G:\\CV\\Dataset\\Detection\\Visdrone\\TEMP\\Visdrone_ship"
res_dir = "G:\\CV\\Dataset\\Detection\\Visdrone\\TEMP\\test_ships"

ship_list = os.listdir(ship_dir)
pool_list = os.listdir(pool_dir)

for i, ship in enumerate(ship_list):
    if i < 0:
        continue
    ship_path = osp.join(ship_dir, ship)
    img = cv2.imread(ship_path)

    paster_name = osp.join(pool_dir, pool_list[i%len(pool_list)])
    paster = cv2.imread(paster_name)
    h, w = paster.shape[:2]
    r = max(h, w) / 50
    new_h, new_w = int(h / r), int(w / r)
    paster = cv2.resize(paster, (new_w, new_h))
    temp = pool_list[i%len(pool_list)].split("_")[0]
    box = [[new_w, new_h, 2*new_w, 2*new_h, int(temp)]]
    # img[1*new_h:2*new_h, 1*new_w:2*new_w] = paster
    b1f0 = np.where(paster > 0, 0, 1)  # >10是为了消除因为resize而在边缘产生的乱像素
    temp = img[1*new_h:2*new_h, 1*new_w:2*new_w]
    img[1*new_h:2*new_h, 1*new_w:2*new_w] = temp * b1f0 + paster

    # show_image(img, np.array(box))
    dom = make_xml(box, ship, (new_h, new_w))
    new_path = osp.join(res_dir, "JPEGImages", ship)
    cv2.imwrite(new_path, img)
    with open(osp.join(res_dir, "Annotations", ship[:-4]+'.xml'), 'w') as f:
        f.write(dom.toprettyxml(indent='\t', encoding='utf-8').decode('utf-8'))

with open(osp.join(res_dir, "ImageSets/Main/ship.txt"), 'w') as f:
    for line in ship_list:
        f.writelines(line.strip() + '\n')
