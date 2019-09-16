# A region proposal network.

"""
 Works cited:
 [1] Simonyan, K. and Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition (2015) arXiv:1409.1556 [cs.CV]
 [2] Ren, et. al. Faster R-CNN (2016)
 [3] RockyXu66 Jupyter notebook
"""

import keras
import math
from keras.applications.vgg16 import VGG16
from keras.initializers import RandomNormal
from keras.layers import Input, Conv2D, TimeDistributed, Dense, Flatten, Dropout
from keras.models import Model
from keras_frcnn import ROIMaxPool
from keras import backend as K

from losses import rpn_reg_loss, rpn_cls_loss, full_model_classifier_loss, full_model_regression_loss

# A tuple of (height, width, channels). Thus data_format needs to be set to 'channels_last' when needed.
img_size = (512, 512, 3)
anchor_box_scales = [32, 64, 128]
anchor_box_ratios = [[1, 1], [1 / math.sqrt(2), 2 / math.sqrt(2)], [2 / math.sqrt(2), 1 / math.sqrt(2)]]
num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)
n_classes = 4000
num_rois = 128

def vgg_base(shape_tuple=(None, None, None)):

    # base network is just VGG16 on ImageNet weights. We extract the last convolutional layer and use it as the "top" of our network.
    vgg_model = VGG16(include_top=False, input_shape=shape_tuple)
    last_conv_layer = vgg_model.get_layer('block5_conv3') 
    new_model = Model(inputs=vgg_model.input, outputs=last_conv_layer.output)
    return new_model # has input built in already

def rpn(shared_layers):
    # here's the RPN layers
    sliding_window = Conv2D(512, kernel_size=(3,3), strides=1, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01), data_format='channels_last', name='intermediate_conv1')(shared_layers.output)

    num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)
    # 4 * num_anchors (x+y+w+h wrt ea. anchor) for reg with (1,1) kernel
    # 2 * num_anchors (p_obj and ~p_obj) for cls, (1,1) kernel
    cls = Conv2D(num_anchors * 2, kernel_size=(1, 1), activation='softmax', kernel_initializer='uniform', name='rpn_objectness_box_cls')(sliding_window)
    reg = Conv2D(num_anchors * 4, kernel_size=(1, 1), activation='linear', kernel_initializer='zero', name='rpn_box_reg')(sliding_window)
    return cls, reg

def classifier_layer(shared_layers, rois, num_rois, n_classes):
    """
        Takes in the base of VGG, which outputs a list of filtered feature maps, the requisite list
        of images for ROIMaxPool.

        Second input is given in @param "rois", which is a list of regions of interest, returned by the rpn_layer.
    """

    roi_pool = ROIMaxPool(num_rois)([shared_layers.output, rois])  # default kernel is (7, 7)

    """
        Remember that output shape is (batch_size, num_rois, kernel_size_0, kernel_size_1, channels). We want to
        apply the same net to each roi -- hence the use of TimeDistributed. 

        Other than that caveat, this is essentially the untrained top FC layers of VGG.
    """

    output = TimeDistributed(Flatten(name='flatten'))(roi_pool)
    output = TimeDistributed(Dense(4096, activation='relu', name='fc4096_1'))(output)
    output = TimeDistributed(Dropout(0.5))(output)
    output = TimeDistributed(Dense(4096, activation='relu', name='fc4096_2'))(output)
    output = TimeDistributed(Dropout(0.5))(output)

    """
        Now for the two "sibling layers..."

        The classifier layer is a Dense layer with softmax activation that classifies each ROI.

        The regressor layer is a Dense layer that uses linear activation to parameterize a (x, y, w, h) tuple encoding each
        ROI indexed by class.
    """

    final_cls = TimeDistributed(Dense(n_classes, activation='softmax', kernel_initializer='zero'), name='dense_categorical_classifier')(output)
    final_reg = TimeDistributed(Dense(4 * (n_classes - 1), activation='linear', kernel_initializer='zero'), name='dense_bbox_regressor')(output)
    # don't forget the -1: we don't do a regression box for the implicit background class!

    return [final_cls, final_reg]


roi_input = Input(shape=(num_rois, 4))
nn_base = vgg_base(shape_tuple=img_size)
rpn_layer = rpn(nn_base)

rpn_model = Model(inputs=nn_base.input, outputs=rpn_layer, name="region_proposal")
rpn_model.compile(optimizer='sgd', loss=[rpn_cls_loss(num_anchors), rpn_reg_loss(num_anchors)])
rpn_model.summary()

classifier = classifier_layer(nn_base, roi_input, num_rois, n_classes)
big_boi_classifier = Model(inputs=[nn_base.input, roi_input], outputs=classifier, name="roi_fast_rcnn_classifier")
big_boi_classifier.compile(optimizer='sgd', loss=[full_model_classifier_loss(), full_model_regression_loss(n_classes)])
big_boi_classifier.summary()

