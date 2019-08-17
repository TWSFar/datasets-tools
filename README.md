按照3:7划分训练数据集和测试数据集
放在训练数据的根目录, train_path代表训练数据的目录, label_path代表标签的目录.
fd 和 fl 分别代表数据和标签, 主要两者格式的区别, 根据需求设置.


txt2voc: 将txt标签转为voc的xml形式


generate_txt: 将voc的annotation生成mmdetection需要的形式

xml2json: 将voc的解释xml转变成coco的json类型