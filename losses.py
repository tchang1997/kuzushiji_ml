import keras
import tensorflow as tf
import numpy as np

from keras import backend as K
from config import Settings

C = Settings()

"""
    Formally, this is the Huber loss, a special form of the smooth L1 norm. It's nice because 
        1) it's super convex
        2) it's continuous, and freakin' differentiable too
        3) it's less sensitive to tiny deviations than the L1, and less sensitive to outliers than the L2, so it's the best of both worlds!

    And this is also what they use in the paper, so we'll go with it.
"""
def _smooth_L1_tensor(x, alpha=1.0, dtype='float32'):
    x_abs = K.abs(x)
    x_bool = K.cast(K.less(x_abs, alpha), dtype)
    smooth_l1_value = (x_bool * 0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5 * alpha) * alpha
    return smooth_l1_value

def rpn_reg_loss(num_anchors, weight=1.0): # weight isn't that big of a deal according to the paper

    """
        This loss function is applied to the output of the RoI regression layer of the RPN, which contains 4 * n_anchors normalized real-valued parameters. Specifically,
        for each anchor at index i, t_i = (x, y, h, w).

        In Ren et. al. (2016), the regression loss for the RPN is given as:

        (1 / n_regressions) * ∑(p*_i * L_reg(t_i, t*_i)) where

        p*_i: ground-truth label of objectness of the box. 0 if negative, 1 if positive. The multiplicative nature of this parameter means that only regression on 
              positive object labels contributes to the loss function. In other words, we do not learn a regression function that maps 
        t_i: a vector representing the parameterized (x, y, h, w) coordinates of each bounding box
        t*_i: a vector representing the ground-truth parameterized coordinates of each bounding box.
        L_reg: Smooth L1 (Huber loss) function.

        Division by this regularization term essentially makes this loss function return a "loss-per-anchor" metric for each image, which, given the disparate numbers
        of valid anchors in each image, makes sense.

        Again, since anchors with non-positive objectness should not contribute to the training objective (denoted by the multiplication by p*_i), we multiply the 
        tensor output of _smooth_L1 with an indicator mask - a tensor of 0s and 1s denoting whether an anchor has a non-positive label (p*_i = 0) or a positive 
        label (p*_i = 1). We already calculate the values of p*_i in preprocessing for all anchors for all images in the training set, so this is fairly convenient.

        For this "masking" technique to work, we need our indicator mask to have the same shape as y_pred. Following kentaroy47's solution, we create a tensor y_true
        that stacks the actual bounding-box parameters (values of t*_i) "atop" the mask (values of p*_i). To be precise, as y_pred has a shape of (batch_size, 
        feature_map_height, feature_map_width, 4 * n_anchors), y_true will have a shape of (batch_size, feature_map_height, feature_map_width, 8 * n_anchors), where:

        y_true[:, :, :, :4 * num_anchors] = the mask - whether or not anchor has a positive label (IoU > 0.7)
        y_true[:, :, :, 4 * num_anchors:] = the values of t*_i, (x, y, h, w)

        Note that for any index j in range(4 * num_anchors), y_true[:, :, :, j] will have the same value, since the same indicator value must be applied to all elements
        of each t*_i. The mask can be constructed in practice via a stack operation: K.stack([P*] * 4).
    """

    def rpn_reg_loss_calc(y_true, y_pred):
        smooth_l1_diff = _smooth_L1_tensor(y_true[:, :, :, 4 * num_anchors:] - y_pred) # this tensor, when summed, will give the smooth L1
        n_reg = K.sum(C._epsilon + y_true[:, :, :, :4 * num_anchors]) # num of anchors correspond to positive object labels
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
        n_cls = K.sum(C._epsilon + y_true[:, :, :, num_anchors:]) # number of "valid" classifications

        # shape of y_pred and y_true will be (batch_size, feature_map_height, feature_map_width, 2 * num_anchors)
        # multiplication by the "bottom half" of y_true makes the loss of invalid anchors 0, as required
        binary_cross = K.sum(y_true[:, :, :, num_anchors:] *  K.binary_crossentropy(y_pred[:, :, :, :num_anchors], y_true[:, :, :, :num_anchors])) 
        return weight * binary_cross / n_cls

    return rpn_cls_loss_calc

def full_model_regression_loss(n_classes, weight=1.0):

    def full_reg_loss_calc(y_true, y_pred):
        diff = y_true[:, :, 4 * n_classes:] - y_pred
        diff_abs = K.abs(diff)
        diff_bool = K.cast(K.less(diff_abs, 1.0), 'float32')
        
        smooth_l1_diff = (diff_bool * 0.5 * diff * diff) + (1 - diff_bool) * (diff_abs - 0.5)
        n_reg = K.sum(C._epsilon + y_true[:, :, :4*n_classes])
        return weight * K.sum(y_true[:, :, :4*n_classes] * smooth_l1_diff) / n_reg

    return full_reg_loss_calc

def full_model_classifier_loss(weight=1.0):

    def full_cls_loss_calc(y_true, y_pred):
        return weight * K.mean(K.categorical_crossentropy(y_true, y_pred))

    return full_cls_loss_calc
