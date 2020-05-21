from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

'''
keras有自带的attention、batch_ctc_loss,ctc_decoder可以用，后面可以改为用这个
'''

# tf.enable_eager_execution()


def gru(units):
    if False:
        pass
    # if tf.test.is_gpu_available():
    #     return tf.keras.layers.CuDNNGRU(units,
    #                                     return_sequences=True,
    #                                     return_state=True,
    #                                     recurrent_initializer='glorot_uniform')
    else:
        return tf.keras.layers.GRU(units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_activation='sigmoid',
                                   recurrent_initializer='glorot_uniform')


class Encoder_raw(tf.keras.Model):
    def __init__(self, enc_units, batch_sz):
        super(Encoder_raw, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.cnn = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, [3, 3], padding="same", activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
            tf.keras.layers.Conv2D(128, [3, 3], padding="same", activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
            tf.keras.layers.Conv2D(256, [3, 3], padding="same", activation='relu'),
            tf.keras.layers.Conv2D(256, [3, 3], padding="same", activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=[2, 1], strides=[2, 1]),
            tf.keras.layers.Conv2D(512, [3, 3], padding="same", activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(512, [3, 3], padding="same", activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=[2, 1], strides=[2, 1]),
            tf.keras.layers.Conv2D(512, [2, 2], strides=[2, 1], padding="same", activation='relu'),
            tf.keras.layers.Reshape((25, 512))
        ])

        self.gru = gru(self.enc_units)

    def call(self, x):
        x = self.cnn(x)
        output, state = self.gru(x)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))




from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

def vgg(input_tensor):
    """
    The original feature extraction structure from CRNN paper.
    Related paper: http://arxiv.org/abs/1507.05717
    """
    x = layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu')(input_tensor)
    x = layers.MaxPool2D(pool_size=2, padding='same')(x)

    x = layers.Conv2D(
        filters=128,
        kernel_size=3,
        padding='same',
        activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, padding='same')(x)

    for i in range(2):
        x = layers.Conv2D(filters=256, kernel_size=3, padding='same',
                          activation='relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='same')(x)

    for i in range(2):
        x = layers.Conv2D(filters=512, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='same')(x)

    x = layers.Conv2D(filters=512, kernel_size=2, activation='relu')(x)
    return x


def vgg16():
    base_model = VGG16(weights='imagenet', include_top=False,pooling=None)
    base_model.trainable = False
    vgg_model = keras.Sequential()

    # 将vgg16模型的 卷积层 添加到新模型中（不包含全连接层)
    ##-5是48可以自己加pooLling层变成24，也可以用默认的到-4是12
    for item in base_model.layers[:-5]:
        vgg_model.add(item)
    vgg_model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='same'))


    '''
    或者是    
    for item in base_model.layers[:-4]:
        vgg_model.add(item)
    '''


    # vgg_model = keras.Sequential([
    #     base_model,
    #     # keras.layers.GlobalAveragePooling2D()
    # ])
    return vgg_model

print('vgg16 summary',vgg16().summary())

def Encoder(enc_units, batch_sz):
    img_input = keras.Input(shape=(32, None, 3))

    # x = vgg(img_input)#self def vgg output x_shape=(None, 1, None, 512)
    x=vgg16()(img_input)
    print('vgg16 output shape',x.shape)#base_model output x_shape= (None, 1, None, 512)


    x = layers.Reshape((-1, 512))(x)

    # x=gru(enc_units)(x)#output, state


    # [x,_,_,_,fh]= layers.Bidirectional(layers.LSTM(units=256, return_sequences=True,return_state=True))(x)
    # # print('lstm x={},c={}.h={}'.format(x.shape,c.shape,h.shape))
    # # print('y',np.array(y).shape,y)
    #
    # [x,_,_,_,fh]= layers.Bidirectional(layers.LSTM(units=256, return_sequences=True,return_state=True))(x)

    #self inputs=Tensor("input_2:0", shape=(None, 32, None, 3), dtype=float32), outputs=Tensor("dense/Identity:0", shape=(None, None, 63), dtype=float32)
    #inputs=Tensor("input_2:0", shape=(None, 32, None, 3), dtype=float32),
    # outputs=Tensor("bidirectional_1/Identity:0", shape=(None, None, 512), dtype=float32)
    # x=(x,fh)

    x= layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(x)
    x= layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(x)

    print('inputs={}, outputs={}'.format(img_input,x))
    return keras.Model(inputs=img_input, outputs=(x,x[:, -1, :]), name='CRNN')



class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 25, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 25, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 25, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(enc_output, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x1 = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x2 = tf.concat([tf.expand_dims(context_vector, 1), x1], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x2)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)

        return x, state, attention_weights
