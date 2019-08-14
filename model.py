#ML
from keras.applications import densenet
from keras.layers import Input, Flatten, Conv2D, MaxPooling2D, LSTM, Reshape, TimeDistributed, Dense, Activation, Permute, Embedding, RepeatVector, Add
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
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

# Settings
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Constants
data_path = "/Users/tchainzzz/python/kuzushiji_ml/"
img_path = data_path + "input/"
OOV = "U+22999"
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
n_classes = len(char_dict)
img_size = (512, 512, 1)
max_seq_length = 256 # max(df_train['labels'].apply(seq_length))
token_unk = 9

def one_hot(seq):
    encoding = list()
    for ch in seq:
        vector = [0 for _ in range(n_classes)]
        vector[df_chars[df_chars['Unicode'] == ch].index[0]] = 1
        encoding.append(vector)
        return np.array(encoding)

def inverse_one_hot(matrix):
    return [np.argmax(vec) for vec in matrix]

def seq_to_vec(seq):
    embedding = []
    for ch in seq: 
        embedding.append(df_chars[df_chars['Unicode'] == ch].index[0])
    if len(embedding) <= max_seq_length:
        embedding += [token_unk] * (max_seq_length - len(embedding))
    elif len(embedding) > max_seq_length:
        embedding = embedding[:max_seq_length]
    if len(embedding) != max_seq_length: print(len(embedding))
    return np.array(embedding)

def vec_to_seq(seq):
    vec = []
    for n in seq:
        if n >= 0:
            vec.append(df_chars.iloc[int(n)]['Unicode'])
    return vec

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
    conv_2d_7 = Conv2D(16, kernel_size=(1, 1), padding="same")(conv_2d_6))
    batch_3 = BatchNormalization()(conv_2d_4)
    max_pool_4 = MaxPooling2D(pool_size=(2, 2), strides=None, padding="same", data_format="channels_last")(batch_3)

    # janky dimensional reduction 
    perm_1 = Permute((3, 2, 1))(max_pool_4)
    reshaper_1 = Reshape((16, -1))(perm_1)
    fc_1 = Dense(max_seq_length)(reshaper_1)
    perm_2 = Permute((2, 1))(fc_1)

    decoder_inputs = Input(shape=(max_seq_length,))
    embedding = Embedding(n_classes, 16)(decoder_inputs)

    decoder_lstm = LSTM(max_seq_length, return_sequences=True)
    concat = Add()([perm_2, embedding])
    decoder_outputs= decoder_lstm(concat)
    time_distr_lstm= TimeDistributed(Dense(n_classes, activation='softmax'))(decoder_outputs)
    

    model= Model(inputs=[input_layer, decoder_inputs], outputs=time_distr_lstm)
    model.compile(Adadelta(rho=0.95, epsilon=1e-8), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

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
            new_arr = np.array(thresh)[:,:,np.newaxis]
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
    np.append(concat, token_unk)
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
decoder_target[max_seq_length] = pd.Series(np.array([token_unk] * len(df_encodings.index)))
print(decoder_target.head())

# train model
ed.fit(train_data, keras.utils.np_utils.to_categorical(decoder_target.to_numpy('int64')), batch_size=8, epochs=100, validation_split=0.2, callbacks=[
    EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10),
    ModelCheckpoint('unfinished_best_model.h5', monitor='val_loss', mode='min', save_best_only=True)    
    ])

# save model
ed.save('kuzushiji_cnn_lstm.h5')

# evaluate model

