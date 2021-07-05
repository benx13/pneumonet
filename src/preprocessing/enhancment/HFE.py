from numpy.fft import fftshift, ifft2, fft2

from src.utils.utils import *


class HFE:

    def __init__(self, d0v=10):
        self.d0v = d0v
        assert 1 <= self.d0v <= 90

    def set_d0v(self, d0v):
        self.d0v = d0v

    def run(self, image):
        """Runs the algorithm for the image."""

        if len(image.shape) > 2:
            image = to_grayscale(image)
        img = normalize(np.min(image), np.max(image), 0, 255, image)

        # HF part

        img_fft = fft2(img)  # img after fourier transformation
        img_sfft = fftshift(img_fft)  # img after shifting component to the center

        (m, n) = img_sfft.shape
        filter_array = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                filter_array[i, j] = 1.0 - np.exp(-((i - m / 2.0) ** 2
                                                    + (j - n / 2.0) ** 2) / (2 * self.d0v ** 2))
        k1 = 0.5
        k2 = 0.75
        high_filter = k1 + k2 * filter_array

        img_filtered = high_filter * img_sfft
        img_hef = np.real(ifft2(fftshift(img_filtered)))  # HFE filtering done

        # HE part
        # Building the histogram

        (hist, bins) = histogram(img_hef)

        # Calculating probability for each pixel

        pixel_probability = hist / hist.sum()

        # Calculating the CDF (Cumulative Distribution Function)

        cdf = np.cumsum(pixel_probability)
        cdf_normalized = cdf * 255
        hist_eq = {}
        for i in range(len(cdf)):
            hist_eq[bins[i]] = int(cdf_normalized[i])

        for i in range(m):
            for j in range(n):
                image[i][j] = hist_eq[img_hef[i][j]]

        return image.astype(np.uint8)
