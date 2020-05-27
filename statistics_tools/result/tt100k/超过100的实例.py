"""
i2' 'i4' 'i5' 'il100' 'il60' 'il80' 'io' 'ip' 'p10' 'p11' 'p12' 'p19'
 'p23' 'p26' 'p27' 'p3' 'p5' 'p6' 'pg' 'ph4' 'ph4.5' 'ph5' 'pl100' 'pl120'
 'pl20' 'pl30' 'pl40' 'pl5' 'pl50' 'pl60' 'pl70' 'pl80' 'pm20' 'pm30'
 'pm55' 'pn' 'pne' 'po' 'pr40' 'w13' 'w32' 'w55' 'w57' 'w59' 'wo']
"""

import numpy as np
classes = np.zeros(1000)
cls = []


with open('statistics_tools\\result\\tt100k\\train_labels_info.txt') as f:
    for i, line in enumerate(f.readlines()):
        if line[0] == '-':
            break
        key, v = line.split(':')
        classes[i] += int(v)
        cls.append(key.strip())

with open('statistics_tools\\result\\tt100k\\test_labels_info.txt') as f:
    for i, line in enumerate(f.readlines()):
        if line[0] == '-':
            break
        key, v = line.split(':')
        classes[i] += int(v)

idx = np.where(classes >= 100, True, False)

print(np.array(cls)[idx[:221]])

pass