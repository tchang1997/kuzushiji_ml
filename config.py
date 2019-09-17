"""
    This is a class that stores global, unmodifiable information like image size and other settings. This should only be modified
    here by hand, and not through code. If you disregard this warning and do anyway, good luck.
"""
import math

class Settings():
    def __init__(self):

        # Main network setup

        self._anchor_box_scales = [32, 64, 128]
        self._img_size = (512, 768, 3)
        self._anchor_box_ratios = [[1, 1], [1 / math.sqrt(2), 2 / math.sqrt(2)], [2 / math.sqrt(2), 1 / math.sqrt(2)]]
        self._num_anchors = len(self._anchor_box_scales) * len(self._anchor_box_ratios)
        self._max_num_rois = 128

        #
        # This depends on neural net structure, but in general, ∏(sqrt(pool area)) over all pooling layers is a good start. 
        # For the shared layers we use from VGG, this is sqrt((2*2)^4) = 16.
        #
        self._rpn_stride = 16

        self._iou_upper = 0.7
        self._iou_lower = 0.3
 
        # Losses

        self._epsilon = 1e-4

