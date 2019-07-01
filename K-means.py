import sklearn as sk
import sklearn.cluster as cluster


def ratio():
    file1 = "result/w_div_h.txt"

    data_ratio = open(file1, 'r')

    data = [[float(line.strip('\n'))] for line in data_ratio]

    clf = cluster.KMeans(n_clusters=3)
    res = clf.fit(data)
    print(clf.cluster_centers_)

def scale():
    file1 = "result/w_h_minset.txt"

    data_ratio = open(file1, 'r')

    data = [[float(line.strip('\n'))] for line in data_ratio]

    clf = cluster.KMeans(n_clusters=3)
    res = clf.fit(data)
    print(res)
    print(clf.cluster_centers_)

def box():
    file1 = "result/boxes_wh.txt"

    data_ratio = open(file1, 'r')

    data = [[float(line.strip('\n').split(' ')[-1])] for line in data_ratio]


if __name__ == "__main__":
    ratio()
    scale()
    box()