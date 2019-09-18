"""
    We need to preprocess our image, bounding-box coordinate, and bounding-box label data in a way that our model understands. The tough part isn't dealing with
    the input, but rather generating proper ground truth labels for everything.
    
    Note that contrary to the format used in Girshick and the other files (in particular, the RoI pooling layer in keras_frcnn.py), OpenCV has width before height.  

"""

import pandas as pd
import numpy as np
import cv2
import random

from collections import Counter
import operator

from pathlib import Path
import pickle

from progress import ProgressTracker # this is a custom module I wrote to track progress when doing work across an iterable
from config import Settings

from enum import Enum, auto, unique

@unique
class Object(Enum):
    POS = auto()
    NONE = auto()
    NEG = auto()


"""
    Utility function for calculating IoU (intersection over union) scores for bounding box overlap. Used to generate ground-truth labels. From RockyXu66's Jupyter notebook.
"""


def iou(a, b):

    def union(au, bu, area_intersection):
        area_a = (au[2] - au[0]) * (au[3] - au[1])
        area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
        area_union = area_a + area_b - area_intersection
        return area_union

    def intersection(ai, bi):
        x = max(ai[0], bi[0])
        y = max(ai[1], bi[1])
        w = min(ai[2], bi[2]) - x
        h = min(ai[3], bi[3]) - y
        if w < 0 or h < 0:
            return 0
        return w*h

    # a and b should be (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)

"""
    Given a tensor representing a mask, and positive and negative indices of the mask tensor to be sampled, this sets randomly sampled indices of the mask tensor to 0.
"""

def suppress_random(mask, pos_indices, neg_indices, max_regions):
    n_pos_anchors = len(pos_indices[0])
    n_neg_anchors = len(neg_indices[0])
    max_regions = C._max_num_rois

    if len(pos_indices[0]) > max_regions // 2: # we want approx. half positive, half negative labels
        suppressed_region_indices = random.sample(range(n_pos_anchors), n_pos_anchors - max_regions // 2)
        mask[pos_indices[0][suppressed_region_indices], pos_indices[1][suppressed_region_indices], pos_indices[2][suppressed_region_indices]] = 0
        n_pos_anchors = max_regions // 2

    if len(neg_indices[0]) + n_pos_anchors > max_regions: # suppress all but n_pos_anchors 
        suppressed_region_indices = random.sample(range(n_neg_anchors), n_neg_anchors - n_pos_anchors)
        mask[neg_indices[0][suppressed_region_indices], neg_indices[1][suppressed_region_indices], neg_indices[2][suppressed_region_indices]] = 0
    return mask

"""
    Helper function for parameterizing targets for regression
"""
def target_calc_helper(box_coordinates, anchor_coordinates):
    gt_box_x_center = (box_coordinates[3] + box_coordinates[0]) / 2
    gt_box_y_center = (box_coordinates[1] + box_coordinates[2]) / 2
    anchor_x_center = (anchor_coordinates[1] + anchor_coordinates[0]) / 2
    anchor_y_center = (anchor_coordinates[2] + anchor_coordinates[3]) / 2
    anchor_width = anchor_coordinates[1] - anchor_coordinates[0]
    anchor_height = anchor_coordinates[3] - anchor_coordinates[2]

    tx = (gt_box_x_center - anchor_x_center) / anchor_width
    ty = (gt_box_y_center - anchor_y_center) / anchor_height
    tw = np.log((box_coordinates[3] - box_coordinates[0]) / anchor_width)
    th = np.log((box_coordinates[2] - box_coordinates[1]) / anchor_height) 
    return [tx, ty, tw, th]

"""
    Helper to scale the coordinates properly.
"""

def generate_scaled_gt_coordinates(bbox_list, old_width, old_height, new_width, new_height):
    # (x, y, h, w) vectors for each bounding box, scaled to the resized dimension
    gt = np.zeros((len(bbox_list), 4))
    for bbox_index, bbox_info in enumerate(bbox_list):
        gt[bbox_index, 0] = int(bbox_info[1], 10) * new_width / float(old_width)
        gt[bbox_index, 1] = int(bbox_info[2], 10) * new_height / float(old_height)
        gt[bbox_index, 2] = int(bbox_info[3], 10) * new_height / float(old_height) + gt[bbox_index, 1]
        gt[bbox_index, 3] = int(bbox_info[4], 10) * new_width / float(old_width) + gt[bbox_index, 0]
    return gt

"""
    From a Series of strings of bounding box info, reshape each element of Series into a (-1, 5) ndarray with implicit columns (label, x, y, h, w) to consolidate
    bounding box info into a list of ndarrays.
"""

def get_bounding_box_info_from_sequence(bounding_box_sequence_list):
    bbox_list = []
    for seq in bounding_box_sequence_list:
        try:
            class_and_box_coordinates = np.array(seq.split(' ')).reshape(-1, 5)
            class_and_box_coordinates[:, 3], class_and_box_coordinates[:, 4] = class_and_box_coordinates[:, 4], class_and_box_coordinates[:, 3].copy()
            bbox_list.append(class_and_box_coordinates)
        except (ValueError, AttributeError) as e:
            """
                Yes, a triple-nested empty array is required. This is because generalized labeling code is going to iterate through each image, and then extract 
                information. Thus, the code will see this as a single image with labels [[]], with single RoI [], with a null label.
            """
            bbox_list.append(np.array([[[]]])) 
    return bbox_list

"""
    Mostly good for debugging and making sure the output is what is expected - sorting the dictionary doesn't really do anything, and if you need to sort it for 
    functionality reasons, a dictionary is probably the wrong idea.
"""

def sort_dict_by_value(d, descending=True):
    return dict(sorted(main_class_counts.items(), key=operator.itemgetter(1), reverse=descending))

"""
    There's another pragmatic point to consider. There are over 4000 classes of objects that need to be recognized; this severely increases the size of the model. Each
    character bounding box is a potential RoI that must be found, regressed, and classified; with thousands of pages of documents one can easily see how this problem
    can grow. Therefore, we can reduce the number of classes by grouping low-frequency classes into a filler "other" class.
"""

def aux_class_pooling(counter, minimum=10, aux_class_name='other'):
    assert minimum > 0
    min_val = minimum
    if minimum <= 1:
        min_val = minimum * sum(counter.values()) 
    new_counts = {class_label: count for class_label, count in counter.items() if count >= minimum}
    other_class = {aux_class_name: sum(dict(set(counter.items()) - set(new_counts.items())).values())}
    new_counts.update(other_class)
    return new_counts


def report_stats(d, aux_class_name='other'):
    print("Number of classes: {}".format(len(d.keys())))
    print("Number of examples: {}".format(sum(d.values())))
    print("Number of auxilliary-class examples: {} ({:0.4f}% of training data)".format(d[aux_class_name], d[aux_class_name]/sum(d.values())))
                

class DataProvider():

    """
        This provides a structured way to access relevant data. From the data CSV file, we perform a train-test split and also reshape training and testing data into the
        format required. 

        Bounding-box coordinates and class labels are provided in this format in a .csv file:

        image_id       | labels
        ===========================================================================================
        <filename>     | <label_0> <x_0> <y_0> <w_0> <h_0> <label_1> <x_1> <y_1> <w_1> <h_1> ... <h_n>

        For convenience, we process, permute, and save the data in the "labels" column as such:

        for each image in image_id:

        label  | x  | y  | w | h
        ===========================
        U+**** | 10 | 10 | 5 | 8 (placeholder values)
        ———————————————————————————
        U+**** | 5  | 18 | 1 | 5
        ——————————————————————————
            ...
            ...
            ...

        Note that the column names are implicit and only provided for convenience. The type of that "table" is an ndarray.

        Fields:
        self.df: the entire CSV in a DataFrame
        self.df_train: a DataFrame of the training data in raw form (filename + string of sequences and bounding box parameters).
        self.df_test: a DataFrame of the test data in raw form (same format as above).
        self.image_bbox_info: a Python list of ndarrays storing a (?, 5)-shape table for each image (in only df_train) encoding the classes and bounding 
                              boxes contained within the image.
        self.class_labels_by_image: a Python list (unflattened) storing only the classes for each image (in only df_train).
        self.class_counts: a dict mapping each class to its numerical frequency.
        self.n_classes: the total number of classes.
        self.class_label_to_int: a dict mapping each class to an integer label.
        self.all_images: a dict mapping each image filename to its info - dimensions and a list of bounding boxes.
    """

    def __init__(self, data_dir = '../', filename='train.csv', image_dir = '../input/', all_data_path = './img_class_bbox_tables.pkl', p_train=0.8, train_sample_seed=42):
        self.df = pd.read_csv(data_dir + filename)
        self.df_train = self.df.sample(frac=p_train, random_state=train_sample_seed).reset_index()
        print(self.df_train.head(n=10))
        self.df_test = self.df.drop(self.df_train.index).reset_index()

        # arrange the data nicely
        self.image_bbox_info = get_bounding_box_info_from_sequence(self.df_train.iloc[:, 2])

        # create dictionaries for class counts + class labels
        self.class_labels_by_image = [single_image_info[:, 0] for single_image_info in self.image_bbox_info]
        
        # the vectorized operation converts a unicode code point to its character representation
        flattened_sequences = np.vectorize(lambda x: chr(int(x[2:], 16)))(np.concatenate(self.class_labels_by_image, axis=None))
        self.class_counts = aux_class_pooling(Counter(flattened_sequences), minimum=10)
        self.n_classes = len(self.class_counts)

        # print(sort_dict_by_value(main_class_counts))
        self.class_label_to_int = dict(zip(self.class_counts.keys(), range(len(self.class_counts.keys()))))
        report_stats(self.class_counts)

        """
            Now that we've gotten some basic information, it's time to  pair image info with the info we want: (filename, original_height, original_width, 
            list(classes and bounding boxes)).
        """

        my_file = Path("./" + all_data_path)
        if my_file.is_file():
            with open(all_data_path, 'rb') as f:
                self.all_images = pickle.load(f)
        else:
            self.all_images = {}
            self._compile_data_for_all_images(image_dir, self.df_train.iloc[:, 1].items())
            with open(all_data_path, 'wb+') as f:
                pickle.dump(self.all_images, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _compile_data_for_all_images(self, image_dir, filename_list):
        file_progress = ProgressTracker(self.df_train.iloc[:, 1])
        file_progress.start()
        for idx, filename in self.df_train.iloc[:, 1].items():
            file_progress.report("Compiling image data:")
            self._compile_data_for_single_image(idx, image_dir, filename)
            file_progress.iteration_done()

    def _compile_data_for_single_image(self, i, image_dir, filename, extension=".jpg"):
        self.all_images[filename] = {}
        curr_img = cv2.imread(image_dir + filename + extension)
        self.all_images[filename]["orig_height"] = curr_img.shape[0] # because, awesomely, OpenCV is width then height. Yay.
        self.all_images[filename]["orig_width"] = curr_img.shape[1]
        curr_img_bbox_and_classes = self.image_bbox_info[i]
        self.all_images[filename]["bbox_and_classes"] = curr_img_bbox_and_classes
        self.all_images[filename]["anchor_labels"] = self.get_image_rpns(curr_img.shape[0], curr_image.shape[1], curr_img_bbox_and_classes, C)


    def get_image_rpns(self, orig_width, orig_height, image_bbox_info, img_dimension_calc_fn, C):
        resized_img_width, resized_img_height = C._img_size[:2]
        fmap_output_width, fmap_output_height = img_dimension_calc_fn(resized_img_width, resized_img_height) 
        num_bboxes = len(image_bbox_info)

        y_is_valid = np.zeros((fmap_output_height, fmap_output_width, C._num_anchors)) # indicator. 1 if IoU > 0.7 or IoU < 0.3. 0 otherwise.
        y_is_obj = np.zeros((fmap_output_height, fmap_output_width, C._num_anchors)) # indicator. 1 if IoU > 0.7, 0 otherwise
        y_rpn_reg = np.zeros((fmap_output_height, fmap_output_width, 4 * C._num_anchors)) # real-valued normalized coordinates
 
        gt_box_coordinates = generate_scaled_gt_coordinates(image_bbox_info, orig_width, orig_height, resized_img_width, resized_img_height)

        # auxilliary structures for mapping each ground-truth bounding box to its best anchor
        best_anchor_for_bbox = -1 * np.ones((num_bboxes, 4)).astype(int) # indexed by bbox number. 2nd dim is (y-loc, x-loc, anchor_scale_idx, anchor_aspect_ratio_idx)
        best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
        best_coordinates_for_bbox = np.zeros((num_bboxes, 4)).astype(int) 
        best_targets_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)
        n_anchors_per_bbox = np.zeros(num_bboxes).astype(int)

        rpn_progressbar = ProgressTracker(range(C._num_anchors * fmap_output_width * fmap_output_height))
        rpn_progressbar.start()

        # for each anchor shape...
        for anchor_area_index, anchor_dim in enumerate(C._anchor_box_scales):
            for anchor_ratio_index, anchor_aspect_ratio in enumerate(C._anchor_box_ratios):

                # calculate anchor dimensions
                anchor_width = anchor_dim * anchor_aspect_ratio[0]
                anchor_height = anchor_dim * anchor_aspect_ratio[1]

                # at every location of the feature map...
                for ix in range(fmap_output_width):
                    # calculate anchor coordinates
                    x_min = C._rpn_stride * (ix + 0.5) - anchor_width / 2
                    x_max = C._rpn_stride * (ix + 0.5) + anchor_width / 2

                    # and discard those out of bounds. The corresponding anchor will thus retain the default (0, 0) label marking it as invalid.
                    if x_min < 0 or x_max > resized_img_width: 
                        rpn_progressbar.skip_iter(fmap_output_height)
                        continue

                    for jy in range(fmap_output_height): 
                        rpn_progressbar.report("Generating ground-truth regions and regression targets for tensor locations:")
                        rpn_progressbar.iteration_done()
                        y_min = C._rpn_stride * (jy + 0.5) - anchor_height / 2
                        y_max = C._rpn_stride * (jy + 0.5) + anchor_height / 2

                        if y_min < 0 or y_max > resized_img_height: # discard out of bounds 
                            continue

                        bbox_type = Object.NEG
                        best_iou_for_loc = 0.0
                        best_reg_target = None

                        """
                            There are two cases where we will use an anchor's regression target values. 
                            1) IoU > 0.7 or whatever threshold is defined
                            2) max IoU w.r.t. all bounding boxes 
                        
                            Usually the bounding box that yields the best IoU for an anchor has IoU > 0.7, but in the case that
                            this doesn't happen, this helps make sure that each anchor has some target. See Ren et. al. (2016) pg. 5 for more details.

                        """
                        for bbox_index in range(gt_box_coordinates.shape[0]):
                            curr_iou = iou([gt_box_coordinates[bbox_index, 0], gt_box_coordinates[bbox_index, 1],
                                gt_box_coordinates[bbox_index, 3], gt_box_coordinates[bbox_index, 2]], [x_min, y_min, x_max, y_max])
                            curr_reg_targets = target_calc_helper(gt_box_coordinates[bbox_index, :], [x_min, x_max, y_min, y_max])

                            """
                                Update our "best-so-far" bookkeeping structures...

                                If at the end of our run, the current bounding box doesn't have an anchor assigned to it, we can assign the best one here, ensuring
                                that each ground-truth box maps to at least one anchor.
                            """
                            if curr_iou > best_iou_for_bbox[bbox_index]:
                                best_anchor_for_bbox[bbox_index] = [jy, ix, anchor_area_index, anchor_ratio_index]
                                best_iou_for_bbox[bbox_index] = curr_iou
                                best_coordinates_for_bbox[bbox_index] = [x_min, x_max, y_min, y_max]
                                best_targets_for_bbox[bbox_index] = curr_reg_targets

                            """
                                Label object as positive and make relevant changes if IoU > 0.7...
                            """
                            if curr_iou > C._iou_upper:
                                bbox_type = Object.POS
                                n_anchors_per_bbox[bbox_index] += 1
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_reg_target = curr_reg_targets
                            elif C._iou_lower < curr_iou <= C._iou_upper:
                                if bbox_type != Object.POS: 
                                    bbox_type = Object.NONE # indeterminate object
                            # else: curr_iou < C._iou_lower, in which case we do not need to do anything - the label is Object.NEG by default.
                           
                            """
                                Update ground-truth arrays for use in loss calculations.
                            """
                            composite_anchor_index = anchor_ratio_index + anchor_area_index * len(C._anchor_box_ratios)
                            y_is_valid, y_is_obj, y_rpn_reg = self._update_gt_labels(bbox_type, y_is_valid, y_is_obj, y_rpn_reg, jy, ix, composite_anchor_index, tgt_val=best_reg_target) 

        """
            One little issue: there is a small chance that a ground-truth box has no anchor assigned to it. This checks for that, and if no anchor is assigned to a
            ground-truth bounding box, it assigns the best one and automatically marks it as valid and positive, setting the regression targets as well.

            Unless the ground-truth box is nowhere near any anchors. That means you have bigger design flaws in your project to deal with.
        """
        y_is_valid, y_is_obj, y_rpn_reg = self._ensure_gt_to_anchor_mapping(y_is_valid, y_is_obj, y_rpn_reg, n_anchors_per_bbox, best_anchor_for_bbox, best_targets_for_bbox, C)

        """
            We know that there are many more negative anchors than positive anchors, because given an image, it's unlikely that there are objects of interest
            that overlap with that many anchored areas.

            To avert this, we sample a certain amount of negative and positive anchors to balance these classes during RPN training, otherwise the binary 
            classification gets real wonky real quick. Our anchor mask array, y_is_valid, makes this super easy. Simply set any value in the mask array to 0 and 
            the corresponding anchor at the corresponding location no longer contributes to the training objective. Thus, for each image, we have approximately 
            balanced classes.
        """
        positive_anchor_indices = np.asarray(np.logical_and(y_is_obj == 1, y_is_valid == 1)).nonzero()
        negative_anchor_indices = np.asarray(np.logical_and(y_is_obj == 0, y_is_valid == 1)).nonzero()
        y_is_valid = suppress_random(y_is_valid, positive_anchor_indices, negative_anchor_indices, C._max_num_rois)

        y_rpn_cls = np.expand_dims(np.concatenate([y_is_obj, y_is_valid], axis=2), axis=0) # so these can be concatenated with the results from other images along axis 0
        y_rpn_reg = np.expand_dims(np.concatenate([np.repeat(y_is_valid, 4, axis=2), y_rpn_reg], axis=2), axis=0)
        print("Regression target tensor shape:",y_rpn_reg.shape)
        print("Classification target tensor shape:",y_rpn_cls.shape)
        return np.copy(y_rpn_cls), np.copy(y_rpn_reg)

    def _update_gt_labels(self, bbox_type, mask, cls, reg, h, w, anchor, tgt_val=None):
        if bbox_type is Object.POS:
            mask[h, w, anchor] = 1
            cls[h, w, anchor] = 1
            reg_array_start_index = len(tgt_val) * anchor
            reg[h, w, reg_array_start_index:reg_array_start_index+4] = tgt_val 
        elif bbox_type is Object.NEG:
            mask[h, w, anchor] = 1
            cls[h, w, anchor] = 0
        else: # Object.NONE
            mask[h, w, anchor] = 0
            cls[h, w, anchor] = 0
        return mask, cls, reg

    def _ensure_gt_to_anchor_mapping(self, mask, cls, reg, n_anchors, best_anchors, best_targets, C):     
        for bbox_index in range(n_anchors.shape[0]):
            if n_anchors[bbox_index] == 0:
                anchors_aweigh = False
                if best_anchors[bbox_index, 0] != -1:
                    anchors_aweigh = True
                    composite_anchor_index = best_anchors[bbox_index, 3] + best_anchors[bbox_index, 2] * len(C._anchor_box_ratios)
                    y_loc = best_anchors[bbox_index, 0]
                    x_loc = best_anchors[bbox_index, 1]
                    mask[y_loc, x_loc, composite_anchor_index] = 1
                    cls[y_loc, x_loc, composite_anchor_index] = 1
                    reg[y_loc, x_loc, 4 * composite_anchor_index:4 * composite_anchor_index + 4] = best_targets[bbox_index, :]
                if not anchors_aweigh:
                    print("Warning: Bounding Box #{} does not have any anchors that overlap. Consider changing anchor sizes.".format(bbox_index))
        return mask, cls, reg

def example_calc_fn(dim0, dim1):
    return (dim0 // 16, dim1 // 16)

def project_anchors_on_image(img, C):
    C._num_anchors


if  __name__ == '__main__':
    d = DataProvider()
    C = Settings()
    test_image_name = d.df_train.iloc[0, 1]
    img_width = d.all_images[test_image_name]["orig_width"]
    img_height = d.all_images[test_image_name]["orig_height"]
    cls, reg = d.get_image_rpns(img_width, img_height, d.image_bbox_info[0], example_calc_fn, C)
