import cv2

from canny_filter import CannyEdgeDetector
from convolutions import *
from fourier import *
from gauss import gaussian_low_pass, gaussian_high_pass

CONVOLUTION = 1
FOURIER = 2
RANK = 3
BILATERAL = 4
MEAN = 'Mean'
MEDIAN = 'Median'
GAUSSIAN = 'Gaussian'
SHARPEN = 'Sharpen'
SOBEL = 'Sobel'
PREWITT = 'Prewitt'
ISOMETRIC = 'Isometric'
SCHARR = 'Scharr'
LAPLACE = 'Laplace'
LOWPASS = 'Low pass'
HIGHPASS = 'Height pass'
EROSION = 'Erosion'
DILATION = 'Dilation'

# MASK
SHARPEN_MASK = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
SOBEL_MASK_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
SOBEL_MASK_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
PREWITT_MASK_X = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
PREWITT_MASK_Y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
ISOMETRIC_MASK_X = np.array([[-1, 0, 1], [-np.sqrt(2), 0, np.sqrt(2)], [-1, 0, 1]])
ISOMETRIC_MASK_Y = np.array([[-1, -np.sqrt(2), -1], [0, 0, 0], [1, np.sqrt(2), 1]])
SCHARR_MASK_X = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])
SCHARR_MASK_Y = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])
LAPLACE_MASK = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])


def mask_fourier(image):
    radius = 32
    mask = np.zeros_like(image)
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2
    center_coordinates = (cx, cy)
    color = (255, 255, 255)
    thickness = -1  # thickness < 0 fill in circle | thickness > 0 fill out circle
    cv2.circle(mask, center_coordinates, radius, color, thickness)
    return mask


def high_pass(image, method_type=CONVOLUTION, label=GAUSSIAN, kernel=SHARPEN_MASK, sigma=15):
    result = np.zeros_like(image)
    if method_type == CONVOLUTION:
        result = convolve(image, kernel)
    elif method_type == FOURIER:
        fft_image = fft2(image)
        fft_shift = fftshift(fft_image)
        if label == GAUSSIAN:
            mask = gaussian_high_pass(sigma, image.shape)
        else:
            mask = mask_fourier(image)
        high_pass_fft = fft_shift * mask
        result = ifft2(fftshift(high_pass_fft))
    return result


def mask_mean(width, height):
    result = np.ones((width, height), dtype=int)
    dimension = width * height
    result = result / dimension
    return result


def low_pass(image, method_type=CONVOLUTION, label=GAUSSIAN, kernel=MEAN, sigma=15, width=3, height=3, ksize=3,
             sigma_color=None):
    result = np.zeros_like(image)
    if method_type == CONVOLUTION:
        kernel = mask_mean(width, height)
        result = convolve(image, kernel)
    elif method_type == FOURIER:
        fft_image = fft2(image)
        fft_shift = fftshift(fft_image)
        if label == GAUSSIAN:
            mask = gaussian_low_pass(sigma, image.shape)
        else:
            mask = mask_fourier(image)
        low_pass_fft = fft_shift * mask
        result = ifft2(fftshift(low_pass_fft))
    elif method_type == RANK:
        result = median_filter(image, ksize)
    elif method_type == BILATERAL:
        result = bilateral_filter(image, ksize, sigma, sigma_color)
    return result


# derivative filter
def canny_filter(image):
    canny = CannyEdgeDetector(image, sigma=25, weak_pixel=75, high_threshold_ratio=0.15)
    noise_reduce, G, non_max_image, double_threshold, hysteresis = canny.detect()

    return hysteresis


def hybrid_filter(high_pass_image, low_pass_image):
    low = low_pass(low_pass_image, FOURIER, sigma=15)
    high = high_pass(high_pass_image, FOURIER, sigma=15)
    return high + low


def padding_array(array, padding_size):
    height = array.shape[0]
    width = array.shape[1]
    new_height = height + 2 * padding_size
    new_width = width + 2 * padding_size

    padded_array = np.zeros((new_width, new_height))

    for i in range(height):
        for j in range(width):
            padded_array[i + padding_size][j + padding_size] = array[i][j]

    return padded_array


def bilateral_filter(image, ksize, sigma_color, sigma_space):
    result = np.zeros_like(image)
    center = ksize // 2
    padding_size = ksize // 2

    gs = np.zeros((ksize, ksize))
    for y in range(ksize):
        for x in range(ksize):
            gs[y, x] = np.exp(-((y - center) ** 2 + (x - center) ** 2) / (2 * sigma_space ** 2))

    padding_image = padding_array(image, padding_size)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            Iq = padding_image[i:i + ksize, j:j + ksize]
            gr = np.exp(-(Iq - image[i, j]) ** 2 / (2 * sigma_color ** 2))
            w = gs * gr
            Ip = np.sum(w * Iq)
            Wp = np.sum(w)
            result[i, j] = Ip / Wp

    return result


def prewitt_filter(image):
    prewitt_x = convolve(image, PREWITT_MASK_X)
    prewitt_y = convolve(image, PREWITT_MASK_Y)
    result = prewitt_x + prewitt_y
    return result


def scharr_filter(image):
    scharr_x = convolve(image, SCHARR_MASK_X)
    scharr_y = convolve(image, SCHARR_MASK_Y)
    result = scharr_x + scharr_y
    return result


def isometric_filter(image):
    iso_x = convolve(image, ISOMETRIC_MASK_X)
    iso_y = convolve(image, ISOMETRIC_MASK_Y)
    result = iso_x + iso_y
    return result


def threshold_filter(image, threshold):
    result = np.zeros_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] > threshold:
                result[y, x] = 255
    return result


# Laplace filter parameter: Sigma = 50 and Threshold = 10
def laplace_filter(image, sigma, threshold):
    gauss_image = low_pass(image=image, method_type=2, sigma=sigma)
    real_image = np.real(gauss_image)
    laplace_image = convolve(real_image, LAPLACE_MASK)
    result = threshold_filter(laplace_image, threshold)
    return result


'''Rank filter'''


def sort(array):
    length = len(array)
    for i in range(0, length):
        for j in range(0, length - i - 1):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]
    return array


def find_min(array):
    minimum = array[0]
    length = len(array)
    for i in range(1, length):
        if minimum > array[i]:
            minimum = array[i]

    return minimum


def find_median(array):
    sort_array = sort(array)
    length = len(sort_array)
    center = int(np.ceil(length / 2))
    return array[center]


def find_max(array):
    maximum = array[0]
    length = len(array)
    for i in range(1, length):
        if maximum < array[i]:
            maximum = array[i]

    return maximum


# Min filter parameter: ksize <= 5
def min_filter(image, ksize):
    result = image
    rows = image.shape[0]
    cols = image.shape[1]
    for y in range(1, rows - ksize):
        for x in range(1, cols - ksize):
            array = []
            for i in range(y, ksize + y):
                for j in range(x, ksize + x):
                    array.append(image[i, j])
            minimum = find_min(array)
            image[y, x] = minimum
    return result


# Median filter parameter: ksize <= 5
def median_filter(image, ksize):
    result = image
    rows = image.shape[0]
    cols = image.shape[1]
    for y in range(1, rows - ksize):
        for x in range(1, cols - ksize):
            array = []
            for i in range(y, ksize + y):
                for j in range(x, ksize + x):
                    array.append(image[i, j])
            center = find_median(array)
            image[y, x] = center
    return result


# Max filter parameter: ksize <= 5
def max_filter(image, ksize):
    result = image
    rows = image.shape[0]
    cols = image.shape[1]
    for y in range(1, rows - ksize):
        for x in range(1, cols - ksize):
            array = []
            for i in range(y, ksize + y):
                for j in range(x, ksize + x):
                    array.append(image[i, j])
            maximum = find_max(array)
            image[y, x] = maximum
    return result


def rank_filter(image, label, ksize=None):
    result = np.zeros_like(image)
    if label == EROSION:
        result = min_filter(image, ksize)
    elif label == MEDIAN:
        result = median_filter(image, ksize)
    elif label == DILATION:
        result = max_filter(image, ksize)
    return result
