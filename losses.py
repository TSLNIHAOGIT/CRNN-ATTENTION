import tensorflow as tf
from tensorflow import keras


class CTCLoss(keras.losses.Loss):
    def __init__(self, logits_time_major=False, blank_index=-1,
                 reduction=keras.losses.Reduction.AUTO, name='ctc_loss'):
        super().__init__(reduction=reduction, name=name)
        self.logits_time_major = logits_time_major
        self.blank_index = blank_index

    def call(self, y_true, y_pred):
        # self.logits_time_major=False
        # self.blank_index=-1




        #y <tf.RaggedTensor
        #self  y_shape=(25, 10),y_pred_shape=(25, 24, 63)
        #keras y_shape=(25, 10),y_pred_shape=(25, 3, 63) 3<标签的长度（如一个标签的位数）（必须>3,等于也不行）
        print('y_shape={},y_pred_shape={}'.format(y_true.shape, y_pred.shape))
        y_true = tf.cast(y_true, tf.int32)
        print('tyep y',type(y_pred),type(y_true))
        logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
        print('logit_length_shape={}'.format(logit_length.shape))

        '''
      args:
      labels: tensor of shape [batch_size, max_label_seq_length] or SparseTensor
      logits: tensor of shape [frames, batch_size, num_labels],
        '''
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=None,
            logit_length=logit_length,
            logits_time_major=self.logits_time_major,

            # ignore_longer_outputs_than_inputs=True,
            blank_index=self.blank_index)
        # print('loss def',loss)
        return tf.reduce_mean(loss)
if __name__=='__main__':
    l=CTCLoss()
    print(type(l),l)
    loss = keras.losses.SparseCategoricalCrossentropy(),
    print(type(loss),loss)
