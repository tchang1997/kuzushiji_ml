from keras.models import load_model, Model
from keras.layers import Input, TimeDistributed, Dense, LSTM # for copying layer

import pandas as pd
import numpy as np
import pickle

import sys

n = int(sys.argv[1])

data_path = "/Users/tchainzzz/python/kuzushiji_ml/"
img_path = data_path + "input/"
image_array_path = "./image_np_array.pkl"
START = 1
END = 2
OOV = 3
max_seq_length = 256 

df_chars = pd.read_csv(data_path + 'unicode_translation.csv')
df = pd.read_csv(data_path + 'train.csv')
print("Loading image information from file...")
with open(image_array_path, "rb") as f:
    imgs = pickle.load(f)
input_image = imgs[n]
input_image = input_image[np.newaxis, :]
target_seq = df.iloc[n]['labels'].split()[::5]
n_classes = len(df_chars) + 3

def print_model(model, title="Model:", sep="=", width=25):
    print(title)
    print(sep * width)
    for layer in model.layers:
        print("{}: {} => {}".format(layer.name, layer.input_shape, layer.output_shape))
    print()



model = load_model("./models/kuzushiji_cnn_lstm_1565827392.348016.h5")
print_model(model, title="Original model:")

encode_and_transform = Model(inputs=model.input, outputs=model.get_layer('multiply_1').output)
print_model(encode_and_transform, title="Encoder:")
decoder_input_h = Input(shape=(max_seq_length,))
decoder_input_c = Input(shape=(max_seq_length,))
decoder_internal_state = [decoder_input_h, decoder_input_c]
decoder_lstm = LSTM.from_config(model.get_layer('lstm_1').get_config())

encoder_output_to_decoder_input = Input(shape=model.get_layer('multiply_1').output_shape[1:])
lstm_outputs, state_h, state_c = decoder_lstm(
    encoder_output_to_decoder_input, initial_state=decoder_internal_state)
decoder_states = [state_h, state_c]
decoder_outputs = model.get_layer('time_distributed_1')(lstm_outputs)
decoder_model = Model([encoder_output_to_decoder_input] + decoder_internal_state, [decoder_outputs] + decoder_states)
print_model(decoder_model, title="Modified decoder:")

# extract encoder & encode image
#image_tensor = encoder.predict(input_image)
# extract and setup decoder

# decoding loop
target = np.full((1, max_seq_length), END)
target[0] = START
decoded = np.array([])
encoder_output = encode_and_transform.predict([input_image, target])
decoder_inner_state = [np.zeros((1, max_seq_length)), np.zeros((1, max_seq_length))]
print("Predicted:")
while True: 
    result, h, c = decoder_model.predict([encoder_output] + decoder_inner_state)
    index_of_max = result[0][-1].argsort()[-1]
    decoded = np.append(decoded, index_of_max)
    sys.stdout.write(df_chars.iloc[int(decoded[-1]) + 3]['char'])
    sys.stdout.flush()
    target = np.append(decoded, np.full((1, max_seq_length), END))[:max_seq_length]
    decoder_inner_state = [h, c]
    if len(decoded) >= max_seq_length or int(decoded[-1]) in [END, OOV]:
        break
print()
print("Expected:")
print(''.join([df_chars.loc[df_chars['Unicode']==char].iloc[0]['char'] for char in target_seq]))
