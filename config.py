"""
    This is a class that stores global, unmodifiable information like image size and other settings. This should only be modified
    here by hand, and not through code. If you disregard this warning and do anyway, good luck.
"""
import math

class Settings():
    def __init__(self):

        # Main network setup

        self._anchor_box_scales = [32, 64, 128]
        self._img_size = (512, 512, 3)
        self._anchor_box_ratios = [[1, 1], [1 / math.sqrt(2), 2 / math.sqrt(2)], [2 / math.sqrt(2), 1 / math.sqrt(2)]]
        self._num_anchors = len(self._anchor_box_scales) * len(self._anchor_box_ratios)
        self._n_classes = 4000
        self._num_rois = 128

        # Losses

        self._epsilon = 1e-4

