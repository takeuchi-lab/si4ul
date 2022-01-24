from si4ul.si import segmentation_si as si
from si4ul.plot import segmentation_si as plot


def thresholding(image_path, result_dir_path, window_size, bias, is_blur, ksize, sigma_x, sigma_y, is_output_regions):
    local_white = si.LocalWhite(image_path, result_dir_path, window_size, bias, is_blur, ksize, sigma_x, sigma_y)
    local_white.segmentation(is_output_regions)
    return local_white


def psegi_thresholding(local_white, sigma):
    return local_white.psegi(sigma)


def plot_histogram(local_white):
    plot.histogram(local_white)


def plot_histogram_region(local_white, display_statistics):
    plot.histogram_region(local_white, display_statistics)
