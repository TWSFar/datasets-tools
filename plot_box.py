import cv2
import numpy as np
import matplotlib as

def show_image(img, labels):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1).imshow(img[:, :, ::-1])
    plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [0, 0, 2, 2, 0]].T, '-')
    plt.show()
    pass


img = cv2.imread('G:\\CV\\Code\\tools\\datasets-tools\\VOC2012\\train\\JPEGImages\\00000001.jpg')

bbox = np.array([[100, 100, 700, 1500]])

show_image(img, bbox)