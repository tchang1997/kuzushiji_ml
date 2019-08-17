#ML
from keras.applications import densenet
from keras.layers import Input, Flatten, Conv2D, MaxPooling2D, LSTM, Reshape, TimeDistributed, Dense, Activation, Permute, Embedding, BatchNormalization, Multiply 
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.optimizers import Adadelta
import keras
import keras.backend as K

# Data processing
import pandas as pd
import numpy as np
import cv2

# File I/O
from tqdm import tqdm
import pickle
from PIL import Image
import os
import sys
import time
import datetime

# Constants
data_path = "/Users/tchainzzz/python/kuzushiji_ml/"
img_path = data_path + "input/"
START = 1
END = 2
OOV = 3
tqdm.pandas(desc="Progress")

def seq_length(seq):
    try:
        return len(np.array(seq.split(' ')).reshape(-1, 5))
    except ValueError:
        return 0
    except AttributeError:
        return 0


df = pd.read_csv(data_path + 'train.csv')
df_train = df.sample(frac=0.8,random_state=42).reset_index()
df_test=df.drop(df_train.index).reset_index()
df_chars = pd.read_csv(data_path + 'unicode_translation.csv')
char_dict = {codepoint: char for codepoint, char in df_chars.values}
n_classes = len(char_dict) + 3
img_size = (512, 512, 1)
max_seq_length = 256 # max(df_train['labels'].apply(seq_length))

def one_hot(seq):
    encoding = list()
    for ch in seq:
        vector = [0 for _ in range(n_classes)]
        try:
            vector[df_chars[df_chars['Unicode'] == ch].index[0] + 3] = 1
        except IndexError:
            vector[OOV] = 1
        encoding.append(vector)
        return np.array(encoding)

def inverse_one_hot(matrix):
    return [np.argmax(vec) for vec in matrix]

def seq_to_vec(seq):
    embedding = [START]
    for ch in seq:
        try:
            embedding.append(df_chars[df_chars['Unicode'] == ch].index[0] + 3) # 3 is the number of special tags: start symbol, end symbol, and OOV
        except IndexError:
            embedding.append(OOV)
    if len(embedding) <= max_seq_length - 1:
        embedding += [OOV] * (max_seq_length - 1 - len(embedding))
    elif len(embedding) > max_seq_length-1:
        embedding = embedding[:max_seq_length-1]
    if len(embedding) != max_seq_length-1: print(len(embedding))
    return np.array(embedding + [END])

def vec_to_seq(seq):
    vec = []
    for n in seq:
        if n >= 0:
            vec.append(df_chars.iloc[int(n)]['Unicode'])
    return vec

"""
    The architecture of this model was largely inspired by the following papers:
    [1] Le, A.D. Clanuwat, T., Kitamoto, A. (2019). A Human-Inspired Recognition System for Pre-Modern Japanese Historical Documents. Via IEEEAccess.

    The optimization algorithm was informed by the following paper:
    [2] Ruder, S. (2017) An overview of gradient descent optimization algorithms.Via arXiv:1609.04747v2.

"""
def build_encoder_decoder():
    input_layer= Input(shape=img_size)

    conv_2d_1= Conv2D(48, kernel_size=(7, 7), strides=1, padding="same")(input_layer)
    max_pool_1= MaxPooling2D(pool_size=(3, 3), strides=2, padding="same", data_format="channels_last")(conv_2d_1)

    conv_2d_2 = Conv2D(16, kernel_size=(3, 3), strides=1, padding="same")(max_pool_1)
    conv_2d_3 = Conv2D(16, kernel_size=(1, 1), padding="same")(conv_2d_2)
    batch_1 = BatchNormalization()(conv_2d_3)
    max_pool_2 = MaxPooling2D(pool_size=(2, 2), strides=None, padding="same", data_format="channels_last")(batch_1)

    conv_2d_4 = Conv2D(16, kernel_size=(3, 3), strides=1, padding="same")(max_pool_2)
    conv_2d_5 = Conv2D(16, kernel_size=(1, 1), padding="same")(conv_2d_4)
    batch_2 = BatchNormalization()(conv_2d_5)
    max_pool_3 = MaxPooling2D(pool_size=(2, 2), strides=None, padding="same", data_format="channels_last")(batch_2)

    conv_2d_6 = Conv2D(16, kernel_size=(3, 3), strides=1, padding="same")(max_pool_3)
    conv_2d_7 = Conv2D(16, kernel_size=(1, 1), padding="same")(conv_2d_6)
    batch_3 = BatchNormalization()(conv_2d_4)
    max_pool_4 = MaxPooling2D(pool_size=(2, 2), strides=None, padding="same", data_format="channels_last")(batch_3)

    # janky dimensional reduction 
    perm_1 = Permute((3, 2, 1))(max_pool_4)
    reshaper_1 = Reshape((16, -1))(perm_1) # 16 is the number of channels coming out of max_pool_4.
    fc_1 = Dense(max_seq_length)(reshaper_1)
    perm_2 = Permute((2, 1))(fc_1)

 
    decoder_inputs = Input(shape=(max_seq_length,))
    embedding = Embedding(n_classes, 16)(decoder_inputs) # match dimensions of CNN output in the embedding to allow merging

    decoder_lstm = LSTM(max_seq_length, return_sequences=True, return_state=True)    
    concat = Multiply()([perm_2, embedding])
    decoder_outputs, lstm_h, lstm_c = decoder_lstm(concat) # we will need to worry about the sequences and states in the inference loop

    # use lstm_h to compute attention in tandem with the feature maps returned by max_pool_4 (batch_size, w, h, c) <op> lstm_h (seq_length, seq_length)
    attention_base = Activation('softmax')(max_pool_4)
    downsample = Reshape((max_seq_length, -1))(attention_base)
    attention_dense = Dense(max_seq_length)(downsample)
    decoder_with_attention = Multiply()([attention_dense, decoder_outputs])

    time_distr_lstm= TimeDistributed(Dense(n_classes, activation='softmax'))(decoder_with_attention)
    

    model= Model(inputs=[input_layer, decoder_inputs], outputs=time_distr_lstm)
    model.compile(Adadelta(rho=0.95, epsilon=1e-8), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

"""
    Images were preprocessed by loading in grayscale followed by binarization. A Hough transform was used to detect and remove some visually linear features deemed extraneous for feature extraction. Methodology informed by:
    [3] Likforman-Sulem, L., Zahour, A., and Taconet, B. (2006). Text Line Segmentation of Historical Documents: a Survey. In  Special Issue on Analysis of Historical Documents, International Journal on Document Analysis and Recognition, Springer.
"""
image_array_path = "./image_np_array.pkl"
def read_img_data(source_df):
    imgs = []
    if not os.path.exists(image_array_path):
        start = time.time()
        for i, row in source_df.iterrows():
            # progress bar
            eta = (len(source_df.index)-(i+1)) / ((i+1)/(time.time() - start))
            sys.stdout.write("\rLoading image ({}/{}), {:.6s}/s, ETA: {:02d}:{:02d}".format(i+1, len(df_train.index), "{:0.4f}".format((i+1)/(time.time() - start)), int(eta // 60) % 60, int(eta) % 60))

            # actual image loading
            curr_img = cv2.imread(img_path + row['image_id'] + ".jpg")
            curr_img = cv2.resize(curr_img, img_size[:2])
            # load in gray
            gray = cv2.cvtColor(curr_img,cv2.COLOR_BGR2GRAY)
        
            #binarize
            ret, thresh = cv2.threshold(gray, 127,255,cv2.THRESH_BINARY_INV)
            new_arr = np.array(thresh)[:,:,np.newaxis] # because CNN expects four dimensional output: (batch_size, h, w, channels); current output is merely (batch_size, h, w)
            imgs.append(new_arr)
        sys.stdout.write("\n")
        print("Saving image data...")
        with open(image_array_path, "wb+") as f:
            pickle.dump(imgs, f)
    else:
        print("Loading image information from file...")
        with open(image_array_path, "rb") as f:
            imgs = pickle.load(f)
    return imgs

def seq_from_dataframe_row(elem):
    try:
        return elem.split()[::5] 
    except AttributeError:
        return [OOV] * max_seq_length
    except IndexError:
        return [OOV] * max_seq_length

def vec_no_pos_from_dataframe_row(elem):
    return seq_to_vec(seq_from_dataframe_row(elem))

def shift_vec_one_earlier(elem):
    concat = np.array(elem[1:])
    np.append(concat, [END])
    return concat

dataframe_with_encodings_path = './df_train_encoded.csv'
def get_encoding_from_dataframe(source_df, save_path=dataframe_with_encodings_path):
    print("Vectorizing unicode sequences...")
    df_encodings = pd.DataFrame(columns=range(max_seq_length))
    start = time.time()
    for i, row in source_df.iterrows():
        # progress bar
        eta = (len(source_df.index)-(i+1)) / ((i+1)/(time.time() - start))
        sys.stdout.write("\rVectorizing line ({}/{}), {:.6s}/s, ETA: {:02d}:{:02d}".format(i+1, len(source_df.index), "{:0.4f}".format((i+1)/(time.time() - start)), int(eta // 60) % 60, int(eta) % 60))
        df_encodings.loc[i] = vec_no_pos_from_dataframe_row(source_df.iloc[i]['labels'])
    sys.stdout.write("\n")
    print("Saving csv...")
    df_encodings.to_csv(dataframe_with_encodings_path)
    return df_encodings

# compile model
ed = build_encoder_decoder()
ed.summary()
print()

# preprocess data
if not os.path.exists(dataframe_with_encodings_path):
    df_encodings = get_encoding_from_dataframe(df_train)
else:
    print("Loading vectorized unicode sequences...")
    df_encodings = pd.read_csv(dataframe_with_encodings_path, header=0, index_col=0)
print(df_encodings.head())
train_data = [read_img_data(df_train), df_encodings.astype('int64')]
print("Shifting vectors one timestep ahead...")
decoder_target = df_encodings.drop(df_encodings.columns[0], axis=1)
decoder_target[max_seq_length] = pd.Series(np.array([OOV] * len(df_encodings.index)))
print(decoder_target.head())

# train model
model_creation_time = time.time()
# [image, numerical_vector_sequence] => [one_hot_sequence_encoding]
ed.fit(train_data, keras.utils.np_utils.to_categorical(decoder_target.to_numpy('int64')), batch_size=8, epochs=100, validation_split=0.2, callbacks=[
    EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10),
    ModelCheckpoint('./models/unfinished_best_model_{}.h5'.format(model_creation_time), monitor='val_loss', mode='min', save_best_only=True),
    CSVLogger('./training_logs/model_{}.csv'.format(model_creation_time))
    ])

# save model
ed.save('./models/kuzushiji_cnn_lstm_{}.h5'.format(model_creation_time))
keras.utils.plot_model(model, to_file='./models/kuzushiji_cnn_lstm_plot_{}.h5')

# evaluate model

