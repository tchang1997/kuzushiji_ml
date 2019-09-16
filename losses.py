import keras
import tensorflow as tf
import numpy as np

from keras import backend as K

epsilon = 1e-4 # added to denominator to prevent divison by zero 

"""
    Formally, this is the Huber loss, a special form of the smooth L1 norm. It's nice because 
        1) it's super convex
        2) it's continuous, and freakin' differentiable too
        3) it's less sensitive to tiny deviations than the L1, and less sensitive to outliers than the L2, so it's the best of both worlds!

    And this is also what they use in the paper, so we'll go with it.
"""
def _smooth_L1(x, alpha=1.0, dtype='float32'):
    x_abs = K.abs(x)
    x_bool = K.cast(K.less(diff_abs, alpha), dtype)
    smooth_l1_value = (x_bool * 0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5 * alpha) * alpha

def rpn_reg_loss(num_anchors, weight=1.0): # weight isn't that big of a deal according to the paper

    def rpn_reg_loss_calc(y_true, y_pred):
        smooth_l1_diff = _smooth_L1(y_true[:, :, :, 4 * num_anchors:] - y_pred)
        n_reg = K.sum(1e-4 + y_true[:, :, :, :4 * num_anchors]) # num of anchors that are legit
        return weight * K.sum(y_true[:, :, :, :4 * num_anchors] * smooth_l1_diff) / n_reg

    return rpn_reg_loss_calc

def rpn_cls_loss(num_anchors, weight=1.0):
   
    """
        This loss function is applied to the output of the final classification layer of the RPN, which contains n_anchors Bernoulli distributions (via softmax) that act
        as a binary classifier.

        In Ren et. al. (2016), the classification loss for the RPN is given as:

        (1 / n_classifications) * ∑(L_cls(p_i, p*_i)) where

        p_i: predicted probability of anchor i (index in mini-batch) being an object
        p*_i: ground-truth label of objectness. 0 if negative (not an object), 1 if positive. Non-positive non-negative anchors do not contribute to the training objective.
        L_cls: log loss (as defined in the paper), equivalent to binary cross-entropy, which we use here.

        Note: n_classifications = mini_batch_size (total number of anchor-boxes compared).

        Recall that the output of the classification layer has shape (feature_map_height, feature_map_width, 2 * n_anchors). By default, n_anchors = 9. This means that
        AT EACH LOCATION of our feature map, the corresponding output tensor looks like this:

        [P(anchor 0 is an object), P(anchor 1 is an object), ..., P(anchor 8 is an object),
         P(anchor 0 is not an object), ... P(anchor 8 is not an object)] 

        or rather, using the notation of the paper:

        [p_0, p_1, ... p_n
         1 - p_0, 1 - p_1, ... 1 - p_n]

        This means that we would only use the "top half" of this output in our loss calculation.

        However, note that anchors of indeterminate objectness do not contribute to the training objective. We need some way to store that. Since y_pred and y_true must have 
        the same shape, our solution is modified from RockyXu66's Jupyter notebook implementation and Kentaro Yoshioka's (Github: kentaroy47) version of encoding this
        information.

        In our generated y_true label (see custom_preprocessing.py for label generation procedure), we have the following:
        
        The "top half": n_anchor indicator variables - 0 if negative (not an object), 1 if positive. This is the definition of p*_i given above.
        The "bottom half": n_anchor indicator variables - 0 if it is non-positive and non-negative (not a valid object), 1 if it has definite objectness. This is NOT p*_i.

        Thus, if y_true[i] = 0 for i >= n_anchors, it should be functionally ignored and not contribute to the count n_classifications.
        """

    def rpn_cls_loss_calc(y_true, y_pred):
        n_cls = K.sum(epsilon + y_true[:, :, :, num_anchors:]) # number of "valid" classifications

        # shape of y_pred and y_true will be (batch_size, feature_map_height, feature_map_width, 2 * num_anchors)
        # multiplication by the "bottom half" of y_true makes the loss of invalid anchors 0, as required
        binary_cross = K.sum(y_true[:, :, :, num_anchors:] *  K.binary_crossentropy(y_pred[:, :, :, :n_anchors], y_true[:, :, :, :num_anchors])) 
        return weight * binary_cross / n_cls

    return rpn_cls_loss_calc

def full_model_regression_loss(n_classes, weight=1.0):

    def full_reg_loss_calc(y_true, y_pred):
        diff = y_true[:, :, 4 * n_classes:] - y_pred
        diff_abs = K.abs(diff)
        diff_bool = K.cast(K.less(diff_abs, 1.0), 'float32')
        
        smooth_l1_diff = (diff_bool * 0.5 * diff * diff) + (1 - diff_bool) * (diff_abs - 0.5)
        n_reg = K.sum(1e-4 + y_true[:, :, :4*n_classes])
        return weight * K.sum(y_true[:, :, :4*n_classes] * smooth_l1_diff) / n_reg

    return full_reg_loss_calc

def full_model_classifier_loss(weight=1.0):

    def full_cls_loss_calc(y_true, y_pred):
        return weight * K.mean(K.categorical_crossentropy(y_true, y_pred))

    return full_cls_loss_calc
