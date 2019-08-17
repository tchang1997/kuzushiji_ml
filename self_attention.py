from keras import backend as K
from keras.layers import Layer, Dense

"""
    The following code for additive attention is largely adapted from the following sources:
    [1] Text Classification Using Keras, https://androidkt.com/text-classification-using-attention-mechanism-in-keras/
    [2] CyberZHG's Github Repository "keras-self-attention", https://github.com/CyberZHG/keras-self-attention/tree/master/keras_self_attention
    [3] codekansas' Github Repository "keras-language-modeling", https://github.com/codekansas/keras-language-modeling/blob/master/keras_models.py
    [4] Tensorflow Tutorial on Image Captioning with Attention, https://www.tensorflow.org/beta/tutorials/text/image_captioning

    as well as the following papers:
    [5] Vinyals, O., Toshev, A., Bengio, S., and Erhan, D. (2015). Show and Tell: A Neural Image Caption Generator. Via arXiv:1411.4555v2.
    [6] Xu, K., Ba, J. L., Kiros, R., Cho, K., Courville, A., Salakhutdinov, R., Zemel, R., Bengio, Y. (2016). Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. Via arXiv:1502.03044v3.
    [7] Cho, K., Courville, A., Bengio, Y. (2015). Describing Multimedia Content using Attention-based Encoder–Decoder Networks. Via arXiv:1507.01053v1.
    [8] Bahdanau, D., Cho, K., Bengio, Y. (2016). Neural Machine Translation by Jointly Learning to Align and Translate. Via arXiv:1409.0473v7.
"""
class AttentionLayer(Layer):
    # This layer "sits" atop the LSTM. It takes in the transformed output of the encoder with its hidden state output: [(batch_size, input_seq_length, lstm_cells), (lstm_cells, )]
    def __init__(self, units):
        super(Layer, self).__init__()
        self.n_units = units
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    """
    @param features: features is a representation of the encoder output.
    @param hidden_state: hidden_state is the hidden state output by the LSTM at a particular time step.
    """
    # is K.reshape needed? 
    def call(self, features, hidden_state):
        # preprocess model
        hidden_state_with_time = K.expand_dims(hidden_state, axis=1)

        # alignment model: tanh of a weighted linear comb. of inputs + dense activation
        score = self.V(K.tanh(self.W1(features) + self.W2(hidden_state)))
        
        # normalize weights
        attn_weights = K.softmax(score, axis=1)

        # generate context: c_i = sum(a_ij, h_i)
        ctx = Lambda(lambda x: K.sum(x, axis=1))(features * attn_weights)
        return ctx, attn_weights

    
    """
    @param input_shape: a 2-tuple containing the shape of the feature parameters and the shape of the hidden_state parameters. 

    """
    def build(self, input_shape):
        super(AttentionLayer, self).build(input_shape)
        feature_dim = input_shape[2]
        if hasattr(self.attention_vec, '_keras_shape'):
            attention_dim = self.attention_vec._keras_shape[1]
        else:
            raise Exception('AttentionLayer shape index 1 not found. Cannot build layer without information about input shape.')

        self.W1 = self.layer.inner_init((feature_dim, self.n_units), name='{}_W1'.format(self.name))
        self.W2 = self.layer.inner_init((feature_dim, self.n_units), name='{}_W2'.format(self.name))
        self.V = self.layer.inner_init((self.n_units, 1)), name='{}_V'.format(self.name))
        self.trainable_weights += [self.W1, self.W2, self.V]
