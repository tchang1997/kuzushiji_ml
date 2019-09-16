"""
    We need to preprocess our image, bounding-box coordinate, and bounding-box label data in a way that our model understands. The tough part isn't dealing with
    the input, but rather generating proper ground truth labels for everything.
    
    Bounding-box coordinates and class labels are provided in this format in a .csv file:

    image_id       | labels
    ===========================================================================================
    <filename>     | <label_0> <x_0> <y_0> <w_0> <h_0> <label_1> <x_1> <y_1> <w_1> <h_1> ... <h_n>

    Note that contrary to the format used in Girshick and the other files (in particular, the RoI pooling layer in keras_frcnn.py), this version has width first 
    instead of height.

    Much shenanigans can be averted via a reshape(-1, 5) operation on the "labels" column, then a permute operation switching the height and width order, which yields a 
    ndarray. For convenience, we'll write this in a data table format:

    for each image in image_id:

    label  | x  | y  | w | h
    ===========================
    U+**** | 10 | 10 | 5 | 8 (placeholder values)
    ———————————————————————————
    U+**** | 5  | 18 | 1 | 5
    ——————————————————————————
        ...
        ...
        ...

    Note that the column names are implicit and only provided for convenience.

    Notice that now, we have a shape of (?, 5). Each "row" in our ndarray corresponds to a bounding box, or, rather, a RoI. Thus, using the num_rois hyperparameter,
    we can standardize the shape to (num_rois, 5). 

    
    There's another pragmatic point to consider. There are over 4000 classes of objects that need to be recognized; this severely increases the size of the model. Each
    character bounding box is a potential RoI that must be found, regressed, and classified; with thousands of pages of documents one can easily see how this problem
    can grow. Therefore, we can reduce the number of classes by grouping low-frequency classes into a filler "other" class.
"""

import pandas as pd

"""
    Utility function for calculating IoU (intersection over union) scores for bounding box overlap. Used to generate ground-truth labels. From RockyXu66's Jupyter notebook.
""" 
def iou(a, b):

    def union(au, bu, area_intersection):
	area_a = (au[2] - au[0]) * (au[3] - au[1])
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
	area_union = area_a + area_b - area_intersection
	return area_union

    def intersection(ai, bi):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
	    return 0
	return w*h

    # a and b should be (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)

"""
    Consider the output of the RPN. The classification layer via a softmax activation results in 9 (number of anchors) Bernoulli distributions
    representing the probability of object-ness at a each spatial location. Therefore, to generate y_true, each image must have a corresponding (dim_0, dim_1, n_anchors * 2) 
    tensor associated with it, where dim_0 and dim_1 are the dimensions of the relevant feature map.

    Ultimately, we need to calculate the IoU for each anchor with respect to each bounding box. Using the ratios specified in Ren et. al. (2016), we have:

    label(anchor) = 
        1 if IoU(anchor, box) > 0.7 for any ground-truth box
        -1 if IoU(anchor, box) < 0.3 for ALL ground-truth boxes
        0    otherwise

    This results in a ground-truth tensor that looks something like this AT EACH SPATIAL LOCATION:
    [1, 0, 0, 0, 0, 0, 0, 1, 0
     0, 1, 1, 1, 1, 1, 1, 0, 1]

     I've made the assumption that we're using 9 anchors, leading to 9 object-ness binary probability distributions.
"""

