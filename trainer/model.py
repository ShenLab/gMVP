import tensorflow as tf


class ModelBase(tf.keras.Model):
    def __init__(self, name=''):
        super(ModelBase, self).__init__(name=name)

    def call(self, inputs, training):
        raise NotImplementedError('The call method has to be override')

    def predict_from_logit(self, logit):
        return tf.nn.sigmoid(logit)
