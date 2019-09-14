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
from keras.layers import Input, Conv2D
from keras.models import Model

from losses import rpn_reg_loss, rpn_cls_loss

# A tuple of (height, width, channels). Thus data_format needs to be set to 'channels_last' when needed.
img_size = (512, 512, 3)
anchor_box_scales = [32, 64, 128]
anchor_box_ratios = [[1, 1], [1 / math.sqrt(2), 2 / math.sqrt(2)], [2 / math.sqrt(2), 1 / math.sqrt(2)]]
num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)

def vgg_base(input_tensor=None):

    input_shape = (None, None, 3)

    # base network is just VGG16 on ImageNet weights. We extract the last convolutional layer and use it as the "top" of our network.
    vgg_model = VGG16(include_top=False, input_shape=input_shape)
    last_conv_layer = vgg_model.get_layer('block5_conv3') 
    new_model = Model(inputs=vgg_model.input, outputs=last_conv_layer.output)
    return new_model # has input built in already

def rpn(shared_layers):
    # here's the RPN layers
    sliding_window = Conv2D(512, kernel_size=(3,3), strides=1, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01), data_format='channels_last', name='intermediate_conv1')(shared_layers.output)

    num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)
    # 4 * num_anchors (x+y+w+h wrt ea. anchor) for reg with (1,1) kernel
    # 2 * num_anchors (p_obj and ~p_obj) for cls, (1,1) kernel
    cls = Conv2D(num_anchors * 2, kernel_size=(1, 1), activation='softmax', kernel_initializer='uniform', name='box_classification')(sliding_window)
    reg = Conv2D(num_anchors * 4, kernel_size=(1, 1), activation='linear', kernel_initializer='zero', name='box_regression')(sliding_window)
    return cls, reg

img_input_shape = (None, None, 3)

nn_base = vgg_base(input_tensor=img_input_shape)
rpn_layer = rpn(nn_base)
rpn_model = Model(nn_base.input, rpn_layer)
rpn_model.compile(optimizer='sgd', loss=[rpn_cls_loss(num_anchors), rpn_reg_loss(num_anchors)])
rpn_model.summary()

