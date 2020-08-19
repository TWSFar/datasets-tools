import numpy as np


def show_image(img, labels):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.plot(labels[:, [0, 0, 2, 2, 0]].T, labels[:, [1, 3, 3,  1, 1]].T, '-')
    # plt.subplot(2, 1, 2).imshow(mask)
    plt.savefig('mask.jpg')
    # plt.show()


def add_mosaic(img_shape, scales=(100, 100), strides=None):
    """img_shape: w, h
    """
    if strides is not None:
        stride_w, stide_h = strides
    else:
        stride_w, stride_h = scales[0] // 2, scales[1] // 2
    num_w, num_h = img_shape[0] // stride_w - 1, img_shape[1] // stride_h - 1
    shift_x = np.arange(0, num_w) * stride_w
    shift_y = np.arange(0, num_h) * stride_h
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel()+scales[0], shift_y.ravel()+scales[1]
    )).transpose().astype(np.int)

    return shifts

img = np.zeros((996, 999, 3))
shifts = add_mosaic((999, 996))

show_image(img, shifts)