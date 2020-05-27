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