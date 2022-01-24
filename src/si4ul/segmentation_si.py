from si4ul.experiments import segmentation_si as exp


def thresholding(image_path, result_dir_path, window_size=50, bias = 1.1, is_blur=True, ksize=(11, 11), sigma_x=0, sigma_y=0, is_output_regions=False):
    """
    local white segmentation algorithm.
    If bias is bigger than 1, algorithm can detect black object. 
    If bias is smaller than 1, algorithm can detect white object. 

    Args:
        image_path (string): image path to segmentation.
        result_dir_path (string): directory to store the segmentation results.
        window_size (int): length of the side of the square in the nearest pixel range.
        bias (float): bias used in threshold determination. If it's bigger than 1, algorithm can detect black object. If it's smaller than 1, algorithm can detect white object. 
        isBlur (bool): whether to use Gaussian smoothing for image preprocessing.
        ksize (Tuple(int, int)): the side of the convolution window for Gaussian smoothing.
        sigma_x (float): standard deviation of x-coordinate for Gaussian smoothing.
        sigma_y (float): standard deviation of y-coordinate for Gaussian smoothing.
        is_output_regions (bool): wheter to make images of each regions.

    Returns:
        LocalWhite: reffer to document of LocalWhite.

    Examples:
        >>> print(segmentation_si.thresholding('./image.jpg', './result/image_18'))
        <si4ul.si.segmentation_si.LocalWhite at 0x10e33efa0>

    """
    if window_size < 2:
        raise ValueError("window_size must be bigger than 2")
    
    return exp.thresholding(image_path, result_dir_path, window_size, bias, is_blur, ksize, sigma_x, sigma_y, is_output_regions)


def psegi_thresholding(local_white, sigma=10):
    """
    post segmentation inference for local white segmentation.

    Args:
        local_white (LocalWhite): segmetation object.
        sigma (float): standard deviation in distribution of input image.

    Returns:
        (float, float, float): test statisitics, PSegI p-value, naive p-value.
        
    Examples:
        >>> print(segmentation_si.psegi_thresholding(local_white))
        (31.08397312484288, 0.08085241645874175, 0.0)

    """
    if sigma < 0:
        raise ValueError("sigma must be bigger than 0")
    return exp.psegi_thresholding(local_white, sigma)


def plot_histogram(local_white):
    """
    plot histogram of pixel values in input image.
    
    Args:
        local_white (LocalWhite): segmetation object.

    """
    exp.plot_histogram(local_white)


def plot_histogram_region(local_white, display_statistics=False):
    """
    plot histogram of pixel values per region.
    
    Args:
        local_white (LocalWhite): segmetation object.

    """
    exp.plot_histogram_region(local_white, display_statistics)