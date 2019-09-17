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
from keras import backend as K

from losses import rpn_reg_loss, rpn_cls_loss, full_model_classifier_loss, full_model_regression_loss
from config import Settings
from roi_max_pool import ROIMaxPool
from labeling import DataProvider

# A tuple of (height, width, channels). Thus data_format needs to be set to 'channels_last' when needed.
C = Settings()

def vgg_base(shape_tuple):

    # base network is just VGG16 on ImageNet weights. We extract the last convolutional layer and use it as the "top" of our network.
    vgg_model = VGG16(include_top=False, input_shape=shape_tuple)
    last_conv_layer = vgg_model.get_layer('block5_conv3') 
    new_model = Model(inputs=vgg_model.input, outputs=last_conv_layer.output)
    return new_model # has input built in already

def rpn(shared_layers):
    # here's the RPN layers
    sliding_window = Conv2D(512, kernel_size=(3,3), padding="same", activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01), data_format='channels_last', name='intermediate_conv1')(shared_layers.output)

    # 4 * num_anchors (x+y+w+h wrt ea. anchor) for reg with (1,1) kernel
    # 2 * num_anchors (p_obj and ~p_obj) for cls, (1,1) kernel
    cls = Conv2D(C._num_anchors * 2, kernel_size=(1, 1), activation='softmax', kernel_initializer='uniform', name='rpn_objectness_box_cls')(sliding_window)
    reg = Conv2D(C._num_anchors * 4, kernel_size=(1, 1), activation='linear', kernel_initializer='zero', name='rpn_box_reg')(sliding_window)
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

        A big bottleneck here is the number of classes. For Kuzushiji classification, n_classes is in excess of 4000. To combat
        this, we aggregate rare classes (those appearing less frequently than some threshold) into an "other" class. Note that the "other"
        class still encodes a valid object. 
    """

    final_cls = TimeDistributed(Dense(data.n_classes, activation='softmax', kernel_initializer='zero'), name='dense_categorical_classifier')(output)
    final_reg = TimeDistributed(Dense(4 * (data.n_classes - 1), activation='linear', kernel_initializer='zero'), name='dense_bbox_regressor')(output)
    # don't forget the -1: we don't do a regression box for the implicit background class!

    return [final_cls, final_reg]

data = DataProvider()

roi_input = Input(shape=(C._max_num_rois, 4))
nn_base = vgg_base(shape_tuple=C._img_size)
rpn_layer = rpn(nn_base)

rpn_model = Model(inputs=nn_base.input, outputs=rpn_layer, name="region_proposal")
rpn_model.compile(optimizer='sgd', loss=[rpn_cls_loss(C._num_anchors), rpn_reg_loss(C._num_anchors)])
rpn_model.summary()

classifier = classifier_layer(nn_base, roi_input, C._max_num_rois, data.n_classes)
big_boi_classifier = Model(inputs=[nn_base.input, roi_input], outputs=classifier, name="roi_fast_rcnn_classifier")
big_boi_classifier.compile(optimizer='sgd', loss=[full_model_classifier_loss(), full_model_regression_loss(data.n_classes)])
big_boi_classifier.summary()

