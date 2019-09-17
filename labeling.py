"""

    We need to preprocess our image, bounding-box coordinate, and bounding-box label data in a way that our model understands. The tough part isn't dealing with
    the input, but rather generating proper ground truth labels for everything.
    
    Note that contrary to the format used in Girshick and the other files (in particular, the RoI pooling layer in keras_frcnn.py), this version has width first 
    instead of height.

"""

import pandas as pd
import numpy as np
import cv2

from collections import Counter
import operator

from pathlib import Path
import pickle

from progress import ProgressTracker

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
    """

    def __init__(self, data_dir = '../', filename='train.csv', image_dir = '../input/', all_data_path = './img_class_bbox_tables.pkl', p_train=0.8, train_sample_seed=42):

        self.df = pd.read_csv(data_dir + filename)
        self.df_train = self.df.sample(frac=p_train, random_state=train_sample_seed).reset_index()
        print(self.df_train.head(n=10))
        self.df_test = self.df.drop(self.df_train.index).reset_index()

        # arrange the data nicely
        self.image_bbox_info = []
        for seq in self.df_train.iloc[:, 2]:
            try:
                class_and_box_coordinates = np.array(seq.split(' ')).reshape(-1, 5)
                class_and_box_coordinates[:, 3], class_and_box_coordinates[:, 4] = class_and_box_coordinates[:, 4], class_and_box_coordinates[:, 3].copy()
                self.image_bbox_info.append(class_and_box_coordinates)
            except (ValueError, AttributeError) as e:

                """
                    Yes, a triple-nested empty array is required. This is because generalized labeling code is going to iterate through each image, and then extract 
                    information. Thus, the code will see this as a single image with labels [[]], with single RoI [], with a null label.
                """

                self.image_bbox_info.append(np.array([[[]]])) 

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
                
        # create dictionaries for class counts + class labels
        self.class_labels_by_image = [single_image_info[:, 0] for single_image_info in self.image_bbox_info]
        

        # the vectorized operation converts a unicode code point to its character representation
        flattened_sequences = np.vectorize(lambda x: chr(int(x[2:], 16)))(np.concatenate(self.class_labels_by_image, axis=None))
        raw_class_counts = Counter(flattened_sequences)
        self.class_counts = aux_class_pooling(raw_class_counts, minimum=10)
        self.n_classes = len(self.class_counts)
        # print(sort_dict_by_value(main_class_counts))
        self.class_label_to_int = dict(zip(self.class_counts.keys(), range(len(self.class_counts.keys()))))
        report_stats(self.class_counts)

        """
            Now that we've gotten some basic information, it's time to
                1) pair image info with the info we want: (filename, original_height, original_width, list(classes and bounding boxes))

        """

        my_file = Path("./" + all_data_path)
        if my_file.is_file():
            with open(all_data_path, 'rb') as f:
                self.all_images = pickle.load(f)
        else:
            file_progress = ProgressTracker(self.df_train.iloc[:, 1])
            self.all_images = {}
            file_progress.start()
            for idx, filename in self.df_train.iloc[:, 1].items():
                file_progress.report("Compiling image data:")
                self.all_images[filename] = {}
                curr_img = cv2.imread(image_dir + filename + ".jpg")
                self.all_images[filename]["orig_height"] = curr_img.shape[0]
                self.all_images[filename]["orig_width"] = curr_img.shape[1]
                self.all_images[filename]["bboxes"] = self.image_bbox_info[idx]
                file_progress.iteration_done()
            
            with open(all_data_path, 'wb+') as f:
                pickle.dump(self.all_images, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(self.all_images[self.df_train.iloc[0, 1]])
        
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
    Consider the output of the RPN. The classification layer via a softmax activation results in 9 (number of anchors) Bernoulli distributions
    representing the probability of object-ness at a each spatial location. Therefore, to generate y_true, each image must have a corresponding (dim_0, dim_1, n_anchors * 2) 
    tensor associated with it, where dim_0 and dim_1 are the dimensions of the relevant feature map.

    Ultimately, we need to calculate the IoU for each anchor with respect to each bounding box. Using the ratios specified in Ren et. al. (2016), we have:

    label(anchor) = 
        1 if IoU(anchor, box) > 0.7 for any ground-truth box
        -1 if IoU(anchor, box) < 0.3 for ALL ground-truth boxes
        0    otherwise

    This results in a ground-truth tensor that looks something like this AT EACH SPATIAL LOCATION:
    [1, 0, 0, 0, 0, 0, 0, 1, 0
     0, 1, 1, 1, 1, 1, 1, 0, 1]

     I've made the assumption that we're using 9 anchors, leading to 9 object-ness binary probability distributions.
"""

if __name__ == '__main__':
    d = DataProvider()
