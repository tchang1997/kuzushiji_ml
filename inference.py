from keras.models import load_model, Model
from keras.layers import Input, TimeDistributed, Dense, LSTM # for copying layer

import pandas as pd
import numpy as np
import pickle

import sys


from keras.preprocessing.sequence import pad_sequences as SequencePadding

n = int(sys.argv[1])

data_path = "/Users/tchainzzz/python/kuzushiji_ml/"
img_path = data_path + "input/"
image_array_path = "./image_np_array.pkl"
START = 4051
END = 4052
max_seq_length = 256 

df_chars = pd.read_csv(data_path + "unicode_translation.csv")
word_index = dict()
with open('./word_index.csv') as f:
    word_index = {int(kv.split(",")[1]): kv.split(",")[0] for kv in list(f)}
df = pd.read_csv(data_path + 'train.csv')
print("Loading image information from file...")
with open(image_array_path, "rb") as f:
    imgs = pickle.load(f)
input_image = imgs[n]
input_image = input_image[np.newaxis, :]
target_seq = df.iloc[n]['labels'].split()[::5]
n_classes = len(word_index) + 2

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

msg = ''.join([df_chars.loc[df_chars['Unicode']==char].iloc[0]['char'] for char in target_seq])


# decoding loop
target_sequence = np.full((1, max_seq_length), 0.0)
target_sequence[0] = START
decoded = np.array([START])
encoder_output = encode_and_transform.predict([input_image, target_sequence])
decoder_inner_state = [np.zeros((1, max_seq_length)), np.zeros((1, max_seq_length))]
print("Predicted:")
curr_idx = 0
while True: 
    result, h, c = decoder_model.predict([encoder_output] + decoder_inner_state) 
    index_of_max = result[0][-1].argsort()[-5:]
    decoded = np.append(decoded, index_of_max)
    if len(decoded) >= max_seq_length or int(decoded[-1]) is END:
         break
    characters_of_max = [df_chars.loc[df_chars['Unicode']==word_index[i].upper()]['char'].iloc[0] for i in index_of_max]
    sys.stdout.write(' '.join(characters_of_max))
    sys.stdout.write("\t")
    sys.stdout.write(msg[curr_idx])
    sys.stdout.write("\n")
    sys.stdout.flush()
    target = np.append(decoded, np.full((1, max_seq_length), END))[:max_seq_length]
    decoder_inner_state = [h, c]
    curr_idx += 1

print()
print("Expected:")
print(msg)
