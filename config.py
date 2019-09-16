"""
    This is a class that stores global, unmodifiable information like image size and other settings. This should only be modified
    here by hand, and not through code. If you disregard this warning and do anyway, good luck.
"""

class Config():
    def __init__(self):
        pass

    @property 
    def _img_size(self):
        return (512, 512, 3)

    @property
    def _num_rois(self):
        return 128
