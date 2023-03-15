import numpy as np

# define mask matrix
# Gaussian + Derivative
SOBEL_MASK_X = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

SOBEL_MASK_Y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])


def convolve(image_matrix, mask):
    if mask.shape[0] != mask.shape[1]:
        assert 'mask should be square'
    if mask.shape[0] % 2 == 0:
        assert 'mask shape should be odd'
    k = mask.shape[0]
    pad_size = k // 2
    image_rows, image_cols = image_matrix.shape
    convolved = np.zeros_like(image_matrix)
    # handle the edge using padding with outside the image with 0
    pad_image = np.pad(image_matrix, pad_size, mode='constant', constant_values=0)

    for i in range(image_rows):
        for j in range(image_cols):
            sub_matrix = pad_image[i:i + k, j:j + k]
            result = np.sum(sub_matrix * mask)

            if result > 255:
                result = 255
            elif result < 0:
                result = 0

            convolved[i, j] = result

    return convolved
