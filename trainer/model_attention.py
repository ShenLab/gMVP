import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from model import ModelBase

from constant_params import window_size

from layers import EvolEncoder2, MultiHeadAttention


class ModelAttention(ModelBase):
    def __init__(self, model_config, name=''):
        super(ModelAttention, self).__init__(name=name)

        print(model_config)
        d_fc = model_config['d_fc']
        d_model = model_config['d_model']
        num_heads = model_config['num_heads']
        attn_score_type = model_config['attn_score_type']
        dropout_rate = model_config.get('dropout_rate', 0.0)

        #self.proj_1d = tf.keras.layers.Dense(d_model, activation='linear')

        act = tf.nn.relu
        self.variant_encoding = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(w, activation=act) for w in [d_fc]])

        self.neighbor_encoding = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(w, activation=act) for w in [d_fc]])

        if model_config['pairwise_type'] != 'none':
            self.pairwise_encoding = tf.keras.models.Sequential(
                [tf.keras.layers.Dense(w, activation=act) for w in [d_fc]])

        #num_species
        #125,exclude fish
        self.evol_encoder = EvolEncoder2(
            num_species=200,
            weighting_schema=model_config['weighting_schema'],
            pairwise_type=model_config['pairwise_type'])

        self.logit_layer = tf.keras.layers.Dense(1)

        self.mha = MultiHeadAttention(
            d_model,
            num_heads,
            attn_score_type=attn_score_type,
            #use_pairwise=model_config['pairwise_type'] != 'none'
            use_pairwise=True)
        self.gru = tf.keras.layers.GRUCell(d_model,
                                           activation='tanh',
                                           recurrent_dropout=dropout_rate,
                                           dropout=dropout_rate)

    def call(self, inputs, training=False, mask=None):
        ref_aa, alt_aa, feature = inputs

        center_pos = window_size // 2
        batch_size = tf.shape(feature)[0]

        seq = tf.one_hot(tf.cast(feature[:, :, 0], tf.int32),
                         depth=20,
                         dtype=tf.float32)

        evol, pairwise = self.evol_encoder(evol[:, :, 21:421])
        context = tf.concat([
            seq,
            evol,
            feature[:, :, 1:21],
            feature[:, :, 421:433],
        ],
                            axis=-1)

        alt_aa = tf.one_hot(tf.cast(alt_aa, tf.int32),
                            depth=20,
                            dtype=tf.float32)
        ref_aa = tf.one_hot(tf.cast(ref_aa, tf.int32),
                            depth=20,
                            dtype=tf.float32)

        center = tf.concat([
            ref_aa, alt_aa, evol[:, center_pos], feature[:, center_pos, 1:21],
            feature[:, center_pos, 421:431]
        ],
                           axis=-1)
        center = self.variant_encoding(center)
        query = center[:, tf.newaxis]

        context = self.neighbor_encoding(context)

        key, value = context, context

        if pairwise is not None:
            pairwise = self.pairwise_encoding(pairwise)

        context = self.mha((query, key, value), pairwise=pairwise, mask=mask)
        x, _ = self.gru(center, context, training=training)

        x = self.logit_layer(x)
        x = tf.squeeze(x, axis=1)

        return x

    def predict(self, inputs, training=False, mask=None):
        logit = self.call(inputs, training, mask)
        return tf.nn.sigmoid(logit)
