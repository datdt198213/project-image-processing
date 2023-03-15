import numpy as np


def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def gaussian_low_pass(sigma, image_shape):
    mask = np.zeros(image_shape[:2])
    rows, cols = image_shape[:2]
    center_coordinate = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            mask[y, x] = np.exp(((-distance((y, x), center_coordinate) ** 2) / (2 * (sigma ** 2))))
    return mask


def gaussian_high_pass(sigma, image_shape):
    mask = np.zeros(image_shape[:2])
    rows, cols = image_shape[:2]
    center_coordinate = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            mask[y, x] = 1 - np.exp(((-distance((y, x), center_coordinate) ** 2) / (2 * (sigma ** 2))))
    return mask
