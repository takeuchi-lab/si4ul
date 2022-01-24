import cv2
import os
import numpy as np
from tqdm import tqdm

from si4ul.si.sicore.sicore import NaiveInferenceNorm, SelectiveInferenceNormSE

class LocalWhite:
    """
    this class returns the results of local white segmentation.
    we can get from segmentation_si.thresholding().
    other segmentation_si APIs need this object to input.

    Attributes:
        bias (float): bias used in threshold determination. If it's bigger than 1, algorithm can detect black object. If it's smaller than 1, algorithm can detect white object. 
        image (array-like of shape (image_height, image_width)): input image of segmentation.
        image_gaussian (array-like of shape (image_height, image_width)): image after Gaussian smoothing.
        image_height (int): height of image.
        image_original (array-like of shape (image_height, image_width)): image matrix read in from the input image.
        image_path (string): image path to segmentation.
        image_width (int): width of image.
        result_dir_path (string): directory to store the segmentation results.
        window_size (int): length of the side of the square in the nearest pixel range.
    
    Examples:
        >>> localWhite.image
        array([[147, 147, 147, ..., 155, 157, 157],
            [147, 147, 148, ..., 154, 156, 156],
            [148, 148, 148, ..., 153, 154, 154],
            ...,
            [159, 158, 157, ..., 152, 151, 150],
            [159, 159, 157, ..., 152, 150, 150],
            [159, 159, 157, ..., 152, 150, 149]], dtype=uint8)
        >>> localWhite.image_height, localWhite.image_width
        (88, 73)
    """
    def __init__(self, image_path, result_dir_path, window_size, bias, is_blur, ksize, sigma_x, sigma_y):
        if not os.path.exists(result_dir_path):
            os.mkdir(result_dir_path)
        
        self.image_path = image_path
        self.result_dir_path = result_dir_path
        self.image_original = cv2.imread(image_path, 0)
        image_original_file_name = os.path.join(result_dir_path, os.path.splitext(os.path.basename(image_path))[0]+'_original.jpg')
        cv2.imwrite(image_original_file_name, self.image_original)

        self.window_size = window_size
        self.bias = bias

        if is_blur:
            self.image_gaussian = cv2.GaussianBlur(self.image_original, ksize, sigma_x, sigmaY=sigma_y, borderType=cv2.BORDER_DEFAULT)
            image_gaussian_file_name = os.path.join(result_dir_path, os.path.splitext(os.path.basename(image_path))[0]+'_gaussian.jpg')
            cv2.imwrite(image_gaussian_file_name, self.image_gaussian)
            self.image = self.image_gaussian
        else:
            self.image = self.image_original
        
        self.image_height, self.image_width = np.shape(self.image)


    def segmentation(self, is_output_regions):
        self.image_segmentation = np.zeros(np.shape(self.image))
        if is_output_regions:
            image_object = np.zeros(np.shape(self.image))
            image_background = np.zeros(np.shape(self.image))
        
        window = np.arange(- int(self.window_size / 2), int(self.window_size / 2) + 1, 1)
        for p_y in tqdm(range(self.image_height)):
            for p_x in range(self.image_width):
                p_sum = 0
                p_count = 0
                for i in window:
                    for j in window:
                        if 0 <= p_y + i < self.image_height and 0 <= p_x + j < self.image_width:
                            p_sum += self.image[p_y + i, p_x + j]
                            p_count += 1
                p_mean = p_sum / p_count
                if p_mean < self.image[p_y, p_x] * self.bias:
                    self.image_segmentation[p_y, p_x] = 255
                    if is_output_regions:
                        image_object[p_y, p_x] = 0
                        image_background[p_y, p_x] = self.image[p_y, p_x]
                else:
                    self.image_segmentation[p_y, p_x] = 0
                    if is_output_regions:
                        image_object[p_y, p_x] = self.image[p_y, p_x]
                        image_background[p_y, p_x] = 0

        image_segmentation_file_name = os.path.join(self.result_dir_path, os.path.splitext(os.path.basename(self.image_path))[0]+'_segmentation.jpg')
        cv2.imwrite(image_segmentation_file_name, self.image_segmentation)
        if is_output_regions:
            image_object_file_name = os.path.join(self.result_dir_path, os.path.splitext(os.path.basename(self.image_path))[0]+'_region1.jpg')
            cv2.imwrite(image_object_file_name, image_object)
            image_background_file_name = os.path.join(self.result_dir_path, os.path.splitext(os.path.basename(self.image_path))[0]+'_region2.jpg')
            cv2.imwrite(image_background_file_name, image_background)


    def psegi(self, sigma):
        n = self.image_height * self.image_width
        vecX = self.image.flatten()

        lower = 0.0
        upper = float('inf')
        count_s = 0.0
        count_t = 0.0
        eta = np.zeros(n)
        image_segmentation_vec = self.image_segmentation.flatten()

        for p in image_segmentation_vec:
            if 0 < p:
                count_s += 1
            else:
                count_t += 1
        if count_s == 0 or count_t == 0:
            return None, lower, upper
        mean_s = 0
        mean_t = 0
        for i, p in enumerate(image_segmentation_vec):
            if 100 < p:
                eta[i] = 1 / count_s
                mean_s += p / count_s
            else:
                eta[i] = -1 / count_t
                mean_t += p / count_t

        var = sigma**2
        naive = NaiveInferenceNorm(vecX, var, eta)
        si = SelectiveInferenceNormSE(vecX, var, eta, init_lower=lower, init_upper=upper)
        c = eta / si.eta_sigma_eta * var
        tau = si.stat

        window = np.arange(- int(self.window_size / 2), int(self.window_size / 2) + 1, 1)
        for p_y in tqdm(range(self.image_height)):
            for p_x in range(self.image_width):
                p_sum = 0
                p_count = 0
                c_sum = 0
                for i in window:
                    for j in window:
                        if 0 <= p_y + i < self.image_height and 0 <= p_x + j < self.image_width:
                            p_sum += self.image[p_y + i, p_x + j]
                            c_sum += c[(p_y + i) * self.image_width + (p_x + j)]
                            p_count += 1
                p_mean = p_sum / p_count
                c_mean = c_sum / p_count
                c_mean -= c[p_y * self.image_width + p_x]
                biased_image_value = self.image[p_y, p_x] * self.bias
                if p_mean < biased_image_value:
                    if c_mean < 0:
                        Lz = -1 * (p_mean - biased_image_value) / c_mean + tau
                        if lower < Lz:
                            lower = Lz
                    elif 0 < c_mean:
                        Uz = -1 * (p_mean - biased_image_value) / c_mean + tau
                        if Uz < upper:
                            upper = Uz
                else:
                    c_mean *= -1
                    if c_mean < 0:
                        Lz = (p_mean - biased_image_value) / c_mean + tau
                        if lower < Lz:
                            lower = Lz
                    elif 0 < c_mean:
                        Uz = (p_mean - biased_image_value) / c_mean + tau
                        if Uz < upper:
                            upper = Uz
        
        naive_p = naive.test()
        selective_p = si.test(intervals=[lower, upper], tail="right", max_dps=50000)
        return tau, selective_p, naive_p
