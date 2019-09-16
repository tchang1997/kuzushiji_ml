# This is an implementation of Faster R-CNN specifically for this domain.
#
# [1] Girshick, R. (2015) Fast R-CNN. IEEE. DOI 10.1109/ICCV.2015.169
# [2] Sevilla, J. (2019) Implementing RoI Pooling in TensorFlow + Keras. xplore.ai on Medium.

import keras
from keras.layers import Layer
from keras import backend as K

import numpy as np
import tensorflow as tf

class ROIMaxPool(Layer):

    """
        
        This is essentially a modified MaxPool2D layer that warps different-dimension feature maps into a constant-dimension
        output space, and takes the place of the final pooling layer in networks such as VGG16. Given some fixed spatial 
        extent H x W, where H and W are hyperparameters, a RoI with dimensions h x w can be subdivided into a grid of
        approximate size h/H x w/W, and then max-pooled within those grid divisions (Girshick 1441). This is particularly useful
        given that object bounding-box sizes can be very different, and this allows a reduction of the feature map space to a
        constant shape. In the end, for each RoI,  there should be an n-length collection of H x W pooled feature maps, where 
        n is the number of channels.
        
        The input shape of this layer has two data inputs: a list of images and a list of RoIs. This means that
        the input is an array of length 2. 

        The list of images therefore has shape (batch_size, img_dim_0, img_dim_1, channels), where img_dim_0 and img_dim_1 
        are the width and height in some order. Girshick places height first. We will follow this convention and place
        height before width whereever possible for the purpose of consistency.

        The list of RoIs has shape (batch_size, n_rois, params), where n_rois are the number of regions of interest that 
        correspond to each image. This parameter must be set at initialization. Similarly, params is a 4-tuple (x, y, h, w) of 
        normalized x-coordinate, normalized y-coordinate, normalized height and normalized width such that all elements of
        the tuple are in [0, 1].

        Similarly, output shape is therefore (batch_size, n_rois, H, W, channels).

        The initial settings for H and W are 7 and 7, as described in Girshick.

        For more information, refer to "Fast R-CNN" by Ross Girshick (arXiv:1504.08083 [cs.CV]).

    """

    def __init__(self, num_rois, pool_h=7, pool_w=7, **kwargs):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.num_rois = num_rois

        super(ROIMaxPool, self).__init__(**kwargs)

    def build(self, input_shape):
        self.n_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        n_channels = input_shape[0][3] # index 0 is the list of images
        return (None, self.num_rois, self.pool_h, self.pool_w, n_channels)

    """
        This is a static method specific to RoI pooling that performs the RoI pool operation on a single image array or feature map.
        It takes in a 2D feature_map, which has shape (dim_0, dim_1), and a region of interest roi, which is a 4-tuple parameterizing
        a region of interest (x, y, h, w). 
    """

    @staticmethod
    def _single_roi_pool(feature_map, roi, pool_height, pool_width):
        assert len(feature_map.shape) == 3
        fmap_h = int(feature_map.shape[0])
        fmap_w = int(feature_map.shape[1])
        x = K.cast(fmap_w * roi[0], 'int32')
        y = K.cast(fmap_h * roi[1], 'int32')
        h = K.cast(fmap_h * roi[2], 'int32')
        w = K.cast(fmap_w * roi[3], 'int32')


        fmap_slice = feature_map[x:x+w, y:y+h, :]

        h_step = K.cast(h / pool_height, 'int32')
        w_step = K.cast(w / pool_width, 'int32')

        subregions = [[(
                        i * w_step, j * h_step, 
                        (i+1) * w_step if i+1 < pool_width else w, (j+1) * h_step if j+1 < pool_height else h
                        ) 
                    for j in range(pool_height)]
                for i in range(pool_height)]

        # Helper that returns the max value in each channel in a region x.
        def pool_area(x):
            return K.max(fmap_slice[x[0]:x[0]+x[3], x[1]:x[1]+x[2], :], axis=[0,1])

        # This is a tensor of the maxima in each subregion in a per-channel basis. Should have shape (H, W, channels).
        pooled_features = K.stack([[pool_area(x) for x in row] for row in subregions])
        return pooled_features

    """
        For a single image, performs maxpool on all RoIs.
    """
    @staticmethod
    def _single_image_rois_pool(feature_map, rois, pooled_height, pooled_width):

        def curried_roi_pool(roi):
            return ROIMaxPool._single_roi_pool(feature_map, roi, pooled_height, pooled_width)

        # This is a tensor that has the results of each single roi_pool along a single axis. Should have shape (num_rois, H, W, channels).
        pooled_areas = K.map_fn(curried_roi_pool, rois, dtype='float32')
        return pooled_areas


    def call(self, input_data, mask=None):
        """
            Because of the mapping operation, we want the nth element of each data_tuple on each call to the curried function. This
            means that passing in input_data directly into K.map_fn is a big no no! Since map_fn operates on dimension 0 of an
            array-like, it will interpret our list of images as the first thing to be processed, then move on to the list of RoIs. 
            Instead, we want map_fn to take things pairwise.
            
            After checking the shape of our input data, we break up data_tuple into our list of images and our list of RoIs:

            img, rois = data_tuple
            
            Now img has shape (batch_size, dim_0, dim_1, channels).
            Also, rois has shape (batch_size, num_rois, params).
        """

        assert len(input_data) == 2
            
        # This is a batch-level roi pooling operation.
        def curried_image_all_rois_pool(data_tuple):
            img, rois = data_tuple
            return ROIMaxPool._single_image_rois_pool(img, rois, self.pool_h, self.pool_w)

        # Should be a tensor with shape (batch_size, self.num_rois, self.pool_h, self.pool_w, channels)
        output = K.map_fn(curried_image_all_rois_pool, input_data, dtype='float32')
        return output
