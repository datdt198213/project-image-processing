import numpy as np

from convolutions import convolve
from fourier import ifft2, fftshift, fft2
from gauss import gaussian_low_pass


class CannyEdgeDetector:
    def __init__(self, image, sigma=15, strong_pixel=255, weak_pixel=75, low_threshold_ratio=0.05,
                 high_threshold_ratio=0.15):
        self.image = image
        self.rows, self.cols = self.image.shape
        self.sigma = sigma
        self.strong_pixel = strong_pixel
        self.weak_pixel = weak_pixel
        self.low_threshold_ratio = low_threshold_ratio
        self.high_threshold_ratio = high_threshold_ratio

    # Noise reduction;
    def noise_reduce(self):
        fft_image = fft2(self.image)
        fft_shift = fftshift(fft_image)
        mask = gaussian_low_pass(self.sigma, self.image.shape)

        low_pass_fft = fft_shift * mask
        result = ifft2(fftshift(low_pass_fft))
        return result

    # Gradient calculation using Sobel
    def sobels(self, smoothed_image):
        Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        Gx = convolve(smoothed_image, Sx)
        Gy = convolve(smoothed_image, Sy)
        # G = sprt(Gx^2+Gy^2)
        G = np.hypot(Gx, Gy)
        G = G / G.max() * 255
        theta = np.arctan2(Gy, Gx)
        return G, theta

    # Non-maximum suppression
    def non_maximum_suppression(self, gradient, theta):
        angle = theta * 180. / np.pi
        processed_matrix = np.zeros_like(gradient, dtype=np.int32)
        for i in range(self.rows - 1):
            for j in range(self.cols - 1):
                q = 255
                r = 255

                # angle 0
                if 0 <= angle[i, j] < 22.5 or 157.5 <= angle[i, j] < 180:
                    q = gradient[i, j + 1]
                    r = gradient[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = gradient[i + 1, j - 1]
                    r = gradient[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = gradient[i + 1, j]
                    r = gradient[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = gradient[i - 1, j - 1]
                    r = gradient[i + 1, j + 1]

                if gradient[i, j] >= q and gradient[i, j] >= r:
                    processed_matrix[i, j] = gradient[i, j]
                else:
                    processed_matrix[i, j] = 0
        return processed_matrix

    # Double threshold
    def double_threshold(self, non_max_image):
        high_threshold = non_max_image.max() * self.high_threshold_ratio
        low_threshold = high_threshold * self.low_threshold_ratio

        result = np.zeros(non_max_image.shape, dtype=np.int32)

        for i in range(self.rows):
            for j in range(self.cols):
                if non_max_image[i, j] >= high_threshold:
                    result[i, j] = self.strong_pixel
                elif low_threshold <= non_max_image[i, j] < high_threshold:
                    result[i, j] = self.weak_pixel

        # strong_i, strong_j = np.where(non_max_image >= high_threshold)
        # weak_i, weak_j = np.where((non_max_image < high_threshold) & (non_max_image >= low_threshold))
        #
        # result[strong_i, strong_j] = self.strong_pixel
        # result[weak_i, weak_j] = self.weak_pixel

        return result

    # Edge Tracking by Hysteresis
    def hysteresis(self, double_threshold_image):
        M, N = double_threshold_image.shape
        weak = self.weak_pixel
        strong = self.strong_pixel
        result = np.copy(double_threshold_image)

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if double_threshold_image[i, j] == weak:

                    if (self.is_strong(double_threshold_image[i + 1, j - 1]) or
                            self.is_strong(double_threshold_image[i + 1, j]) or
                            self.is_strong(double_threshold_image[i + 1, j + 1]) or
                            self.is_strong(double_threshold_image[i, j - 1]) or
                            self.is_strong(double_threshold_image[i, j + 1]) or
                            self.is_strong(double_threshold_image[i - 1, j - 1]) or
                            self.is_strong(double_threshold_image[i - 1, j]) or
                            self.is_strong(double_threshold_image[i - 1, j + 1])):
                        result[i, j] = strong
                    else:
                        result[i, j] = 0

        return result

    def is_strong(self, value):
        return value == self.strong_pixel

    def detect(self):
        noise_reduce = self.noise_reduce()
        G, theta = self.sobels(np.real(noise_reduce))
        non_max_image = self.non_maximum_suppression(G, theta)
        double_threshold = self.double_threshold(non_max_image)
        hysteresis = self.hysteresis(double_threshold)

        return noise_reduce, G, non_max_image, double_threshold, hysteresis
