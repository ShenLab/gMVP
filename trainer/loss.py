import tensorflow as tf


def compute_loss(label, logit):

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit)
    return tf.reduce_mean(loss)
