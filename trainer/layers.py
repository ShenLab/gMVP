import tensorflow as tf

from constant_params import window_size


class EvolEncoder2(tf.keras.layers.Layer):
    def __init__(self,
                 num_species=200,
                 pairwise_type='none',
                 weighting_schema='spe'):
        super(EvolEncoder2, self).__init__()

        self.num_species = num_species
        self.pairwise_type = pairwise_type
        self.weighting_schema = weighting_schema

    def build(self, input_shape):
        if self.weighting_schema == 'spe':
            self.W = self.add_weight(
                name='species_weights',
                shape=(1, self.num_species),
                initializer=tf.keras.initializers.Constant(0.0),
                trainable=True)
            '''
            self.B = self.add_weight(
                name='b',
                shape=[],
                initializer=tf.keras.initializers.Constant(0.0),
                trainable=False)
            '''
        elif self.weighting_schema == 'none':
            self.W = tf.constant(1.0 / self.num_species,
                                 shape=(1, self.num_species),
                                 dtype=tf.float32)
        else:
            raise NotImplementedError(
                f'weighting_schema {weighting_schema} NotImplementedError')

    def call(self, x):
        shape = tf.shape(x)
        B, L, N = shape[0], shape[1], shape[2] // 2
        center_pos = window_size // 2
        A = 21 + 1  #alphabet size
        A = 21

        ww = x[:, :, 200:]
        w = x[:, 0, 200:]
        x = x[:, :, :200]
        if self.weighting_schema == 'spe':
            #W = self.B * w / 100.0 + self.W  #+ tf.cast(tf.less(w, 0.01),
            #          tf.float32) * -1e12
            W = tf.nn.softmax(self.W, axis=-1)
        else:
            W = self.W

        #tf.print(self.B, self.W)
        #x = tf.where(tf.less(ww, 0.01), 21.0, x)
        x = tf.one_hot(tf.cast(x, tf.int32), depth=A, axis=-1)

        #W, (batch, 1, 1, species)
        #x, (batch, len, species, 21)
        #output, (batch, len, 1, 21)
        x1 = tf.matmul(W[:, tf.newaxis, tf.newaxis], x)

        #pairwise
        if self.pairwise_type == 'fre':
            x2 = tf.matmul(x[:, center_pos:center_pos + 1, :, :, tf.newaxis],
                           x[:, :, :, tf.newaxis])
            x2 = tf.reshape(x2, (B, L, N, A * A))
            x2 = tf.matmul(W, x2)
            x2 = tf.squeeze(x2, axis=2)
        elif self.pairwise_type == 'cov':
            #numerical stability
            #(batch_size, window_size, 192, A)
            x2 = x - x1
            x2 = tf.sqrt(W[:, tf.newaxis, :, tf.newaxis]) * x2
            x2 = tf.transpose(x2, perm=(0, 2, 1, 3))
            x2_t = tf.reshape(x2, shape=(B, N, L * A))
            #left(batch_size, 192, A)
            #right(batch_size, 192, L * A)
            #result(batch-size, A, L * A)
            x2 = tf.matmul(x2[:, :, center_pos], x2_t, transpose_a=True)
            x2 = tf.reshape(x2, (B, A, L, A))
            x2 = tf.transpose(x2, (0, 2, 1, 3))
            x2 = tf.reshape(x2, (B, L, A * A))
            norm = tf.sqrt(
                tf.reduce_sum(tf.square(x2), keepdims=True, axis=-1) + 1e-12)
            x2 = tf.concat([x2, norm], axis=-1)
        elif self.pairwise_type == 'cov_all':
            #(batch, len, species, 21)
            x2 = x - x1
            #(batch, species, len, 21)
            x2 = tf.transpose(x2, perm=(0, 2, 1, 3))
            #(batch, species, len * 21)
            x2 = tf.reshape(x2, shape=(B, N, L * A))
            x2 = tf.sqrt(W[:, :, tf.newaxis]) * x2
            x2 = tf.matmul(x2, x2, transpose_a=True)
            x2 = tf.reshape(x2, (B, L, A, L, A))
            x2 = tf.transpose(x2, perm=(0, 1, 3, 2, 4))
            x2 = tf.reshape(x2, (B, L, L, A * A))
            norm = tf.sqrt(
                tf.reduce_sum(tf.square(x2), keepdims=True, axis=-1) + 1e-12)
            x2 = tf.concat([x2, norm], axis=-1)
        elif self.pairwise_type == 'inv_cov':
            x2 = x - x1
            x2 = tf.transpose(x2, perm=(0, 2, 1, 3))
            x2 = tf.reshape(x2, shape=(B, N, L * A))
            x2 = tf.sqrt(W[:, :, tf.newaxis]) * x2
            x2 = tf.matmul(x2, x2, transpose_a=True)
            x2 += tf.eye(L * A) * 0.01
            x2 = tf.linalg.inv(x2)
            x2 = tf.reshape(x2, (B, L, A, L, A))
            x2 = tf.transpose(x2, perm=(0, 1, 3, 2, 4))
            x2 = x2[:, center_pos]
            x2 = tf.reshape(x2, (B, L, A * A))

        elif self.pairwise_type == 'none':
            x2 = None
        else:
            raise NotImplementedError(
                f'pairwise_type {self.pairwise_type} not implemented')

        x1 = tf.squeeze(x1, axis=2)

        return x1, x2


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,
                 d_model,
                 num_heads,
                 attn_score_type='add',
                 use_pairwise=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.out_proj = tf.keras.layers.Dense(d_model)

        self.attn_score_type = attn_score_type

        self.use_pairwise = use_pairwise
        self.use_kv_bias = True
        self.use_position_bias = True

        if self.use_pairwise:
            self.w_pairwise = tf.keras.layers.Dense(d_model)

    def build(self, input_shape):

        if self.use_position_bias:
            self.position_bias = self.add_weight(
                name='position_bias',
                shape=(1, 1, 1, window_size),
                dtype=tf.float32,
                trainable=True,
                initializer=tf.keras.initializers.Zeros())

        if self.attn_score_type == 'nn':
            self.agg_v = self.add_weight(
                name='agg_v',
                shape=(1, self.num_heads, 1, self.depth * 3),
                dtype=tf.float32,
                initializer=tf.keras.initializers.GlorotUniform(),
                trainable=True)

        if self.attn_score_type == 'add':
            self.agg_v = self.add_weight(
                name='agg_v',
                shape=(1, self.num_heads, 1, self.depth),
                dtype=tf.float32,
                initializer=tf.keras.initializers.GlorotUniform(),
                trainable=True)

        if self.use_pairwise and self.attn_score_type == 'dot':
            self.agg_pairwise = self.add_weight(
                name='agg_pairwise',
                shape=(1, self.num_heads, 1, self.depth),
                dtype=tf.float32,
                initializer=tf.keras.initializers.GlorotUniform(),
                trainable=True)
        '''
        self.attention_bias = self.add_weight(
            name='attention_bias',
            shape=(1, self.num_heads, 1, 1),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Zeros(),
            trainable=True)
        '''

        if self.use_kv_bias:
            self.bias_k = self.add_weight(
                name='bias_k',
                shape=(1, 1, self.d_model),
                initializer=tf.keras.initializers.GlorotUniform())
            self.bias_v = self.add_weight(
                name='bias_v',
                shape=(1, 1, self.d_model),
                initializer=tf.keras.initializers.GlorotUniform())

            if self.use_position_bias:
                self.bias_position_bias = self.add_weight(
                    name='bias_position_bias',
                    shape=(1, 1, 1, 1),
                    initializer=tf.keras.initializers.Zeros())

            if self.use_pairwise:
                self.bias_pairwise = self.add_weight(
                    name='bias_pairwise',
                    shape=(1, 1, self.d_model),
                    initializer=tf.keras.initializers.GlorotUniform())

    def call(self, x, pairwise=None, mask=None):
        q, k, v = x
        if self.use_position_bias:
            if self.use_kv_bias:
                position_bias = tf.concat(
                    [self.position_bias, self.bias_position_bias], axis=3)
            else:
                position_bias = self.position_bias

            #position_bias = tf.tile(self.position_bias,
            #                        [1, self.num_heads, 1, 1])

        if self.use_pairwise:
            assert (pairwise is not None)

        shape = tf.shape(k)
        batch_size, neighbor_size = shape[0], shape[1]

        #q, k, v = x1d[:, :1], x1d, x1d

        #set query only for center position
        q = self.wq(q)  # (batch_size, 1, d_model)
        k = self.wk(k)  # (batch_size, neighbor_size, d_model)
        v = self.wv(v)  # (batch_size, neighbor_size, d_model)

        #if self.use_position_bias:
        #    position_bias = 0.5 * (position_bias +
        #                           tf.reverse(position_bias, axis=(3, )))

        if self.use_pairwise:
            pairwise = self.w_pairwise(pairwise)

        if self.use_kv_bias:
            #add bias along the sequence
            k = tf.concat([k, tf.tile(self.bias_k, (batch_size, 1, 1))],
                          axis=1)  #(batch_size, tgt_len + 1, embed_dim)
            v = tf.concat([v, tf.tile(self.bias_v, (batch_size, 1, 1))],
                          axis=1)  #(batch_size, tgt_len + 1, embed_dim)
            if mask is not None:
                mask = tf.pad(mask, paddings=[[0, 0], [0, 1]])

            if self.use_pairwise:
                pairwise = tf.concat(
                    [
                        pairwise,
                        tf.tile(self.bias_pairwise, (batch_size, 1, 1))
                    ],
                    axis=1)  #(batch_size, tgt_len + 1, embed_dim)

        #input=(batch_size, len, embed_dim)
        #output=(batch_size, num_heads, len, depth)
        def _split_heads(x):
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
            x = tf.transpose(x, (0, 2, 1, 3))
            return x

        v = _split_heads(v)

        #scaled-dot attention
        if self.attn_score_type == 'dot':
            q = _split_heads(q)
            k = _split_heads(k)

            #(batch_size, num_heads, 1, neighbor_size)
            attention_logits = tf.matmul(q, k, transpose_b=True)

            if self.use_pairwise:
                #dot_pairwies (1,num_hedas, 1, depth)
                #pairwise(batch_size, num_heads, seq_len, depth)
                pairwise = _split_heads(pairwise)
                pairwise = tf.nn.tanh(pairwise)
                pairwise_attention = tf.matmul(self.agg_pairwise,
                                               pairwise,
                                               transpose_b=True)
                #batch_size, seq_len, num_heads
                attention_logits += pairwise_attention

        elif self.attn_score_type == 'nn':
            q = _split_heads(q)
            k = _split_heads(k)
            s = [tf.tile(q, [1, 1, tf.shape(k)[2], 1]), k]
            if self.use_pairwise:
                pairwise = _split_heads(pairwise)
                s.append(pairwise)
            s = tf.concat(s, axis=-1)  #(batch_size, num_heads, len, depth*3)
            s = tf.matmul(self.agg_v, s,
                          transpose_b=True)  #(batch_size, num_heads, 1, len)
            attention_logits = tf.nn.leaky_relu(s)

        elif self.attn_score_type == 'add':  #additive attention
            '''
            s = q + k
            if self.use_pairwise:
                s += pairwise
            s = tf.nn.tanh(s)
            s = _split_heads(s)
            attention_logits = tf.matmul(self.agg_v, s, transpose_b=True)
            '''
            s = q + k
            if self.use_pairwise:
                s += pairwise
            s = _split_heads(s)
            s = tf.matmul(self.agg_v, s, transpose_b=True)
            attention_logits = tf.nn.leaky_relu(s)
            #attention_logits = tf.nn.tanh(s)

        else:
            raise NotImplementedError(
                f'{attn_score_type} not NotImplementedError')

        attention_logits /= tf.math.sqrt(tf.cast(self.depth, tf.float32))

        if self.use_position_bias:
            attention_logits += position_bias

        attention_logits = tf.squeeze(attention_logits, axis=2)

        #mask, (batch_size, 1, neighbor_size)
        if mask is not None:
            attention_logits += (mask[:, tf.newaxis] * -1e12)

        attention_weights = tf.nn.softmax(
            attention_logits,
            axis=-1)  # (batch_size, num_heads,  neighbor_size)

        #(batch_size,num_heads,1,depth)
        attention_weights = attention_weights[:, :, tf.newaxis]

        scaled_attention = tf.squeeze(tf.matmul(
            attention_weights, v), axis=2)  # (batch_size, num_heads,  depth)

        concat_attention = tf.reshape(
            scaled_attention,
            (batch_size, self.d_model))  # (batch_size, d_model)

        output = self.out_proj(concat_attention)
        #output = concat_attention
        return output
