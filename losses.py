import keras
import tensorflow as tf

from keras import backend as K


def rpn_reg_loss(num_anchors, weight=1.0): # weight isn't that big of a deal according to the paper

    def rpn_reg_loss_calc(y_true, y_pred):
        diff = y_true[:, :, :, 4 * num_anchors:] - y_pred # diff is vanilla L1 between the predicted regressed coordinates
        diff_abs = K.abs(diff) # vanilla L1
        diff_bool = K.cast(K.less(diff_abs, 1.0), tf.float32)
        smooth_l1_diff = (diff_bool * 0.5 * diff * diff) + (1 - diff_bool) * (diff_abs - 0.5) # indicator bool lets us essentially apply smooth L1 element-wise across the vanilla L1 differences
        n_reg = K.sum(1e-4 + y_true[:, :, :, :4 * num_anchors]) # num of anchors that are legit
        return weight * K.sum(y_true[:, :, :, :4 * num_anchors] * smooth_l1_diff / n_reg)

    return rpn_reg_loss_calc

def rpn_cls_loss(num_anchors, weight=1.0):
    
    def rpn_cls_loss_calc(y_true, y_pred):
        n_cls = K.sum(1e-4 + y_true[:, :, :, :2*num_anchors])
        binary_cross = K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, :2*num_anchors]) 
        return weight * binary_cross / n_cls

    return rpn_cls_loss_calc
