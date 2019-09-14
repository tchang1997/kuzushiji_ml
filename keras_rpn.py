# A region proposal network.

"""
 Works cited:
 [1] Simonyan, K. and Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition (2015) arXiv:1409.1556 [cs.CV]
 [2] Ren, et. al. Faster R-CNN (2016)
"""

import keras
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Conv3D

# A tuple of (height, width, channels). Thus data_format needs to be set to 'channels_last' when needed.
img_size = (512, 512, 3)

def build_model():
    vgg_model = VGG16(include_top=False, input_shape=img_size)
    last_conv_layer = vgg_model.get_layer('block5_conv3')
    roi_sliding_window = Conv3D(512, kernel_size=(3,3), strides=1, data_format='channels_last')(last_conv_layer)
    return roi_sliding_window

nn = build_model()
nn.compile(optimizer='adam', loss='categorical_crossentropy')
nn.summary()

