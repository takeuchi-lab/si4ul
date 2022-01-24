# SI_FOR_UNSUPERVISED_LEARNING

---
# CLUSTERING

## K-means

### kmeans_si.kmeans
```
kmeans_si.kmeans(X, n_clusters, max_iter = 1000, random_seed = 0)
```

K-means clustering algorithm.

Parameters:
- X: array-like of shape (n_samples, n_features)
    - The observations to cluster. 
- n_clusters: int
    - The number of clusters.
- max_iter: int
- random_seed: int

Returns: 
- centroid: array-like of shape (n_clusters, n_features)
- label: array-like of shape (n_samples)


### kmeans_si.pci_gene
```
kmeans_si.pci_gene(X, comp_cluster, n_clusters, obs_model, max_iter=1000, seed=0, Var=1)
```

Parameters:
- X: array-like of shape (n_samples, n_features)
    - The observations to cluster. 
- comp_cluster: array-like of shape (n_clusters)
- n_clusters: int
    - The number of clusters.
- obs_model: object of K-means clustering result
- max_iter: int
- random_seed: int
- Var: int

Returns: 
- stat: float
- selective_p_value: float
- naive_p_value: float
- sigma: float



### kmeans_si.pci_cluster
```
kmeans_si.pci_cluster(X, comp_cluster, n_clusters, obs_model, max_iter=1000, seed=0, Var=1)
```

K-means clustering algorithm.

Parameters:
- X: array-like of shape (n_samples, n_features)
    - The observations to cluster. 
- comp_cluster: array-like of shape (n_clusters)
- n_clusters: int
    - The number of clusters.
- obs_model: object of K-means clustering result
- max_iter: int
- random_seed: int
- Var: int

Returns: 
- stat: float
- selective_p_value: float
- naive_p_value: float
- sigma: float


---
# CHANGE_POINT_DETECTION

---
# SEGMENTATION

## API

### segmentation_si.thresholding
```
segmentation_si.thresholding(image_path, result_dir_path, window_size=50, bias = 1.1, isBlur=True, ksize=(11, 11), sigma_x=0, sigma_y=0, is_output_regions=False)
```

local white segmentation algorithm.
If bias is bigger than 1, algorithm can detect black object. 
If bias is smaller than 1, algorithm can detect white object. 

Parameters:
- image_path: string
    - image path to segmentation
- result_dir_path: string
    - directory to store the segmentation results
- window_size: int
    - length of the side of the square in the nearest pixel range
- bias: float
    - bias used in threshold determination. If it's bigger than 1, algorithm can detect black object. If it's smaller than 1, algorithm can detect white object. 
- isBlur: boolean
    - whether to use Gaussian smoothing for image preprocessing
- ksize: Tuple(int, int)
    - the side of the convolution window for Gaussian smoothing
- sigma_x: float
    - standard deviation of x-coordinate for Gaussian smoothing
- sigma_y: float
    - standard deviation of y-coordinate for Gaussian smoothing
- is_output_regions: boolean
    - wheter to make images of each regions.

Returns: 
- local_white: LocalWhite
    - segmentation object


### segmentation_si.psegi_thresholding
```
segmentation_si.psegi_thresholding(local_white, sigma=10)
```

post segmentation inference for local white segmentation.

Parameters:
- local_white: LoaclWhite
    - segmetation object
- sigma: float
    - standard deviation in distoribution of input image

Returns: 
- naive_p: float
    - p_value for z-test
- selective_p: float
    - p_value for selective inference


### segmentation_si.plot_histogram
```
segmentation_si.plot_histogram(local_white)
```

plot histogram of pixel values in input image.

Parameters:
- local_white: LoaclWhite
    - segmetation object


## segmentation_si.plot_histogram_region
```
segmentation_si.plot_histogram_region(local_white, display_statistics=False)
```

plot histogram of pixel values per region.

Parameters:
- local_white: LoaclWhite
    - segmetation object
- display_statistics: boolean
    - wheter display arrow that means test statistics


## Object

### segmentation_si.LocalWhite

this class returns the results of local white segmentation.
we can get from `segmentation_si.thresholding()`.
other segmentation_si APIs need to input.

Variables:
- image_path: string
    - image path to segmentation
- result_dir_path: string
    - directory to store the segmentation results
- image_original: array-like of shape (image_height, image_width)
    - image matrix read in from the input image
- window_size: int
    - length of the side of the square in the nearest pixel range
- bias: float
    - bias used in threshold determination. If it's bigger than 1, algorithm can detect black object. If it's smaller than 1, algorithm can detect white object. 
- image_gaussian: array-like of shape (image_height, image_width)
    - image after Gaussian smoothing
- image: array-like of shape (image_height, image_width)
    - input image of segmentation
- image_height: int
    - height of image
- image_width: int
    - width of image
