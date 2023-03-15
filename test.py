import time
import unittest

import cv2
import numpy as np
from matplotlib import pyplot as plt

import gauss
from canny_filter import CannyEdgeDetector
from convolutions import convolve
from filter import hybrid_filter, low_pass, high_pass
from fourier import fft, fft2, ifft, ifft2, fftshift


class TestFunction(unittest.TestCase):
    def test_canny(self):
        image = cv2.imread('images/girl.png', cv2.IMREAD_GRAYSCALE)

        canny = CannyEdgeDetector(image, sigma=25, weak_pixel=75, high_threshold_ratio=0.15)
        noise_reduce, G, non_max_image, double_threshold, hysteresis = canny.detect()

        fig, axis = plt.subplots()
        plt.subplot(2, 3, 1), plt.imshow(image, cmap='gray')
        plt.title('origin')

        plt.subplot(2, 3, 2), plt.imshow(np.real(noise_reduce), cmap='gray')
        plt.title('smooth')

        plt.subplot(2, 3, 3), plt.imshow(G.astype(np.int), cmap='gray')
        plt.title('sobel gradient')

        plt.subplot(2, 3, 4), plt.imshow(non_max_image, cmap='gray')
        plt.title('non maximum suppression')

        plt.subplot(2, 3, 5), plt.imshow(double_threshold, cmap='gray')
        plt.title('double threshold')

        plt.subplot(2, 3, 6), plt.imshow(hysteresis, cmap='gray')
        plt.title('hysteresis')

        tittle = "sigma: " + str(canny.sigma) + " - weak pixel: " + str(
            canny.weak_pixel * canny.low_threshold_ratio) + " - high pixel: " + str(
            canny.strong_pixel * canny.high_threshold_ratio)

        plt.suptitle(tittle, fontweight="bold")
        plt.subplots_adjust(top=1.1)
        fig.tight_layout()
        plt.show()

    def test_hybrid(self):
        high_pass_image = cv2.imread('images/boy.png', cv2.IMREAD_GRAYSCALE)
        low_pass_image = cv2.imread('images/girl.png', cv2.IMREAD_GRAYSCALE)
        hybrid_image = hybrid_filter(high_pass_image, low_pass_image)

        fig, axis = plt.subplots()
        plt.subplot(1, 3, 1), plt.imshow(high_pass_image, cmap='gray')
        plt.title('einstein')

        plt.subplot(1, 3, 2), plt.imshow(low_pass_image, cmap='gray')
        plt.title('marilyn')

        plt.subplot(1, 3, 3), plt.imshow(np.real(hybrid_image), cmap='gray')
        plt.title('hybrid image')

        fig.tight_layout()
        plt.show()

    def test_low_pass_fourier(self):
        low_pass_image = cv2.imread('images/girl.png', cv2.IMREAD_GRAYSCALE)
        low = low_pass(image=low_pass_image, method_type=2, sigma=15)
        plt.imshow(np.real(low), cmap='gray')
        plt.title('Low Pass Fourier Filter')
        plt.show()

    def test_high_pass_fourier(self):
        high_pass_image = cv2.imread('images/girl.png', cv2.IMREAD_GRAYSCALE)
        high = high_pass(image=high_pass_image, method_type=2, sigma=15)
        plt.imshow(np.real(high), cmap='gray')
        plt.title('High Pass Fourier Filter')
        plt.show()

    def test_convolve(self):
        img = cv2.imread('images/femme.png', cv2.IMREAD_GRAYSCALE)
        row, column = img.shape
        test_mask = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        output_image_lib = cv2.filter2D(src=img,
                                        ddepth=-1,
                                        kernel=test_mask)

        output = convolve(img, test_mask)
        expected_value = output[1:row - 2, 1:column - 2]
        actual_value = output_image_lib[1:row - 2, 1:column - 2]
        self.assertTrue((expected_value == actual_value).all())

    def test_fourier_1d(self):
        test = np.random.random(1024)
        start = time.time()
        f2 = np.fft.fft(test)
        end = time.time()
        print('np.fft.fft: ', (end - start) * 1000, 'ms')

        start = time.time()
        f1 = fft(test)
        end = time.time()
        print('fft_recursive: ', (end - start) * 1000, 'ms')
        self.assertTrue(np.allclose(f1, f2))

        ifft_recursive = ifft(f1)
        self.assertTrue(np.allclose(test, ifft_recursive))

    def test_fourier_2d(self):
        test_2d = cv2.imread('images/images.jpg', cv2.IMREAD_GRAYSCALE)
        start = time.time()
        fft2_recursive = fft2(test_2d)
        end = time.time()
        print('fft2_recursive: ', (end - start) * 1000, 'ms')

        start = time.time()
        fft2_np = np.fft.fft2(test_2d)
        end = time.time()
        print('np.fft.fft2: ', (end - start) * 1000, 'ms')
        self.assertTrue(np.allclose(fft2_recursive, fft2_np))

        shift = fftshift(fft2_recursive)
        shift_np = np.fft.fftshift(fft2_np)
        self.assertTrue(np.allclose(shift, shift_np))

        ifft2_np = np.fft.ifft2(fft2_np)
        ifft2_r = ifft2(fft2_recursive)
        self.assertTrue(np.allclose(ifft2_np, ifft2_r))

        # show visualization
        fig, axis = plt.subplots()
        plt.subplot(3, 2, 1), plt.imshow(50 * np.log(abs(fft2_recursive)), cmap='gray')
        plt.title('fft')

        plt.subplot(3, 2, 3), plt.imshow(50 * np.log(abs(shift)), cmap='gray')
        plt.title('shift fft')

        plt.subplot(3, 2, 5), plt.imshow(abs(ifft2_r), cmap='gray')
        plt.title('inverse fft')

        plt.subplot(3, 2, 2), plt.imshow(50 * np.log(abs(fft2_np)), cmap='gray')
        plt.title('fft numpy')

        plt.subplot(3, 2, 4), plt.imshow(50 * np.log(abs(shift_np)), cmap='gray')
        plt.title('shift numpy')

        plt.subplot(3, 2, 6), plt.imshow(abs(ifft2_np), cmap='gray')
        plt.title('inverse fft numpy')

        fig.tight_layout()
        plt.show()

    def test_gauss(self):
        gauss_low = gauss.gaussian_low_pass(1, (5, 5))
        # image = cv2.imread('images/girl.png', cv2.IMREAD_GRAYSCALE)
        #
        # fig, axis = plt.subplots()
        # gauss_matrix = gauss.gaussian_low_pass(1, (7, 7))
        # plt.subplot(2, 3, 1), plt.imshow(gauss_matrix, cmap='gray')
        # plt.title('7x7')
        # plt.imshow(gauss_matrix, cmap='gray')
        #
        # low = low_pass(image=image, sigma=1)
        #
        # gauss_matrix = gauss.gaussian_low_pass(1, (5, 5))
        # plt.subplot(2, 3, 2), plt.imshow(gauss_matrix, cmap='gray')
        # plt.title('5x5')
        # plt.imshow(gauss_matrix, cmap='gray')
        #
        # gauss_matrix = gauss.gaussian_low_pass(1, (3, 3))
        # plt.subplot(2, 3, 3), plt.imshow(gauss_matrix, cmap='gray')
        # plt.title('3x3')
        # plt.imshow(gauss_matrix, cmap='gray')
        #
        # fig.tight_layout()
        # plt.show()
