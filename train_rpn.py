import keras_rpn
from config import Settings
import pickle

C = Settings()

if __name__ == '__main__':

    # build model
    nn_base = keras_rpn.vgg_base(shape_tuple=C._img_size)
    rpn_model = keras_rpn.build_rpn(nn_base, verbose=True)

    # load labeled data
    with open("./rpn_cls.pkl", "rb") as f:
        cls = pickle.load(f)
    with open("./rpn_reg.pkl", "rb") as f:
        reg = pickle.load(f)

    rpn_model.train(
    
