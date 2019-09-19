import numpy as np

"""
    Helper function for parameterizing targets for regression
"""
def target_calc_helper(box_coordinates, anchor_coordinates):
    cdef float gt_box_x_center = (box_coordinates[3] + box_coordinates[0]) / 2
    cdef float gt_box_y_center = (box_coordinates[1] + box_coordinates[2]) / 2
    cdef float anchor_x_center = (anchor_coordinates[1] + anchor_coordinates[0]) / 2
    cdef float anchor_y_center = (anchor_coordinates[2] + anchor_coordinates[3]) / 2
    cdef float anchor_width = anchor_coordinates[1] - anchor_coordinates[0]
    cdef float anchor_height = anchor_coordinates[3] - anchor_coordinates[2]

    cdef float tx = (gt_box_x_center - anchor_x_center) / anchor_width
    cdef float ty = (gt_box_y_center - anchor_y_center) / anchor_height
    cdef float tw = np.log((box_coordinates[3] - box_coordinates[0]) / anchor_width)
    cdef float th = np.log((box_coordinates[2] - box_coordinates[1]) / anchor_height) 
    return [tx, ty, tw, th]

"""
    Utility function for calculating IoU (intersection over union) scores for bounding box overlap. Used to generate ground-truth labels. From RockyXu66's Jupyter notebook.
"""


cdef int union(au, bu, area_intersection):
    cdef int area_a = (au[2] - au[0]) * (au[3] - au[1])
    cdef int area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    cdef int area_union = area_a + area_b - area_intersection
    return area_union

cdef int intersection(ai, bi):
    cdef int x = max(ai[0], bi[0])
    cdef int y = max(ai[1], bi[1])
    cdef int w = min(ai[2], bi[2]) - x
    cdef int h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w*h

def iou(a, b):
    # a and b should be (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    cdef int area_i = intersection(a, b)
    cdef int area_u = union(a, b, area_i)

    return area_i / (area_u + 1e-6)
