import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.filters import gaussian
from scipy.signal import find_peaks
from scipy.signal import convolve2d
from bias_triangle_detection.danon_gap import extract
from dataclasses import dataclass

@dataclass
class DanonGapDetector:
    method: str = "MF"
    min_kernel_size: int = 3
    max_kernel_size: int = 21
    sigma: float = 1
    prominence: float = 0.5
    prior_n: int = 4
    detection_threshold: float = 0.85
    field_gap_size: float = 0.02
    relative_depth: float = 1.3
    plot: bool = False
    peak_offset_tolerance: float = 0.025

    def get_danon_gap_kernel(self, kernel_size_height = 10, kernel_size_width = 6):
        kernel = np.ones((kernel_size_height, kernel_size_width))
        # kernel[:, int(kernel_size / 2)] = np.ones(kernel_size).T
        # kernel[int(kernel_size / 2), :] = np.ones(kernel_size)
        x = np.linspace(-1,1,kernel_size_width)
        normal = np.atleast_2d(np.exp(-x**2/0.5**2))
        kernel = kernel*normal
        return kernel/np.sum(kernel)

    def get_prior(self,array,axis=0):
        domain = np.linspace(-array.shape[axis]/2,array.shape[axis]/2,array.shape[axis])
        width = array.shape[axis]/2
        #flat top Gaussian distribution
        normal = np.exp(-(domain/width)**self.prior_n)
        return normal[:,np.newaxis]

    def get_gap_location(self, data, return_figure = True):
        if self.method == "MF":
            return self.mf_gap_location(data, return_figure)
        elif self.method == "OU":
            return self.ou_gap_location(data.T, return_figure)
        raise NotImplementedError(f"Method {self.method} is not supported, please select one between MF or OU")
    
    def mf_gap_location(self, data, return_figure):
        step = np.diff(data[data.dims[0]])[0]
        field_gap_size_in_pixels = int(np.round(self.field_gap_size/step))
        y = extract(data,
                    sigma = self.sigma,
                    field_gap_size = field_gap_size_in_pixels,
                    relative_depth = self.relative_depth)
        if self.plot or return_figure:
            fig = plt.figure()
            plt.imshow(data)
            plt.colorbar()
            plt.title("Gap NOT detected")
            if y is not None:
                plt.axhline(y=y, c="red", linestyle = "dashed")
                plt.title("Gap detected")
        if return_figure:
            return y, fig
        else:
            return y

    def ou_gap_location(self, data, return_figure):
        kernel_size_height = data.shape[0]//2

        # Smooth data
        data = (data - data.min()) / (data.max() - data.min())
        smoothed = gaussian(data, sigma=self.sigma)

        peak_mask = np.zeros_like(smoothed)
        peaks_x, peaks_y = [], []
        std = []
        for idx in range(smoothed.shape[0]):
            # Choose slice of data
            trace = np.abs(smoothed[idx])
            trace = (trace - trace.min()) / (trace.max() - trace.min())

            # Find peaks in data
            peaks, _ = find_peaks(trace, prominence=self.prominence)

            peaks_x.append(peaks)
            peaks_y.append([idx] * len(peaks))

            for p in peaks:
                peak_mask[idx, p] = 1

        #kernel = self.get_kernel(kernel_size=11)
        #plt.figure()
        #plt.imshow(kernel)
        #plt.show()
        # Repeat scoring for multiple kernel sizes
        score_mask = np.zeros_like(peak_mask)
        # for kernel_size in np.arange(self.min_kernel_size,self.max_kernel_size+1,2):
        #     kernel = self.get_kernel(kernel_size=kernel_size)
        #     line_filter = convolve2d(peak_mask, kernel, mode='same')
        #     score_mask = score_mask + line_filter
        for kernel_size in np.arange(self.min_kernel_size,self.max_kernel_size+1,2):
            kernel = self.get_danon_gap_kernel(kernel_size_height=kernel_size_height+1,
                                               kernel_size_width=kernel_size)
            line_filter = convolve2d(peak_mask, kernel, mode='same')
            score_mask = score_mask + line_filter

        #Normalise score_mask to be between 0 and 1
        score_mask = score_mask / np.max(score_mask)
        # Prior knowledge that there is typically noise around the top/bottom edges
        prior = self.get_prior(score_mask, axis=0)
        score = score_mask * prior

        score_thresh = score.copy()
        score_thresh[score < self.detection_threshold] = 0
        readout_point = np.unravel_index(np.argmax(score_thresh), np.array(score_thresh).shape)
        if readout_point[0] == 0 and readout_point[1] == 0:
            readout_point = [None, None]

        if self.plot or return_figure:
            fig, axs = plt.subplots(4,1, figsize=(5, 10))
            axs[0].imshow(data)
            # axs[1].colorbar()
            axs[0].grid(False)
            axs[0].set_title('original')

            axs[1].imshow(score)
            # axs[1].colorbar()
            axs[1].grid(False)
            axs[1].set_title('line filter')


            axs[2].imshow(score_thresh)
            # axs[2].colorbar()
            axs[2].grid(False)
            axs[2].set_title('thresholded line filter')

            axs[3].imshow(data)
            # axs[3].colorbar()
            axs[3].grid(False)
            axs[3].set_title('gap point')
            axs[3].scatter(readout_point[1], readout_point[0], color='r', marker='o')
            plt.tight_layout()
            if self.plot:
                plt.show()

        if return_figure:
            return readout_point[1], fig
        else:
            return readout_point[1]