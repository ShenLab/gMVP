import tensorflow as tf
import os
import pickle
import numpy as np

from constant_params import input_feature_dim, window_size


def build_dataset(input_tfrecord_files, batch_size):
    drop_remainder = False

    feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'ref_aa': tf.io.FixedLenFeature([], tf.int64),
        'alt_aa': tf.io.FixedLenFeature([], tf.int64),
        'feature': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
        'var_id': tf.io.FixedLenFeature([], tf.string),
    }

    def _parser(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        label, ref_aa, alt_aa = parsed['label'], parsed['ref_aa'], parsed[
            'alt_aa']
        var_id = parsed['var_id']

        ref_aa, alt_aa, label = tf.cast(ref_aa, tf.int32), tf.cast(
            alt_aa, tf.int32), tf.cast(label, tf.float32)

        feature = tf.io.decode_raw(parsed['feature'], tf.float32)
        feature = tf.reshape(feature, (window_size, input_feature_dim))

        mask = tf.io.decode_raw(parsed['mask'], tf.float32)
        mask = tf.reshape(mask, (window_size, ))
        h = window_size // 2

        #mask the postion of interest
        mask = tf.concat(
            [mask[:h],
             tf.cast([
                 1,
             ], dtype=tf.float32), mask[h + 1:]],
            axis=-1)
        '''
        pos_encoding = 1.0 + tf.cast(
            tf.math.abs(window_size // 2 - tf.range(window_size)),
            dtype=tf.float32)
        #pos_encoding = tf.math.log() / tf.math.log(2.0)
        feature = tf.concat([feature, pos_encoding[:, tf.newaxis]], axis=-1)
        '''

        return var_id, ref_aa, alt_aa, feature, label, mask

    dataset = tf.data.TFRecordDataset(input_tfrecord_files)

    options = tf.data.Options()
    options.experimental_threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)

    dataset = dataset.shuffle(2048)
    dataset = dataset.map(_parser, num_parallel_calls=8)

    dataset = dataset.batch(batch_size)

    #dataset = dataset.prefetch(4)

    return dataset


def build_all_possible_missenses_dataset(tr_list, feature_dir, batch_size):
    amino_acid_order = 'ACDEFGHIKLMNPQRSTVWY*'

    def _gen_data():
        for transcript_id in tr_list:
            feature_path = f'{feature_dir}/{transcript_id}.pickle'
            if not os.path.exists(feature_path):
                continue
            print(feature_path, flush=True)

            with open(feature_path, 'rb') as fr:
                feature = pickle.load(fr)

            L = feature.shape[0]

            w = window_size // 2

            for aa_pos in range(L):
                ref_aa = int(feature[aa_pos, 0])

                start = max(aa_pos - w, 0)
                end = min(L, aa_pos + 1 + w)
                var_start = start - (aa_pos - w)
                var_end = var_start + (end - start)
                var_feature = np.zeros([w * 2 + 1, feature.shape[1]])
                var_feature[var_start:var_end] = feature[start:end]

                mask = np.ones((w * 2 + 1, ), dtype=np.float32)
                mask[var_start:var_end] = 0.0
                mask[w] = 1.0

                for alt_aa in range(20):
                    var_id = f'{transcript_id}_{str(aa_pos+1)}_{amino_acid_order[ref_aa]}_{amino_acid_order[alt_aa]}'.encode(
                        'utf-8')
                    yield var_id, np.int32(ref_aa), np.int32(
                        alt_aa), np.float32(var_feature), np.float32(mask)

    dataset = tf.data.Dataset.from_generator(
        _gen_data, (tf.string, tf.int32, tf.int32, tf.float32, tf.float32),
        (tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape(
            ()), tf.TensorShape((window_size, input_feature_dim)),
         tf.TensorShape((window_size, ))))

    options = tf.data.Options()
    options.experimental_threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)

    #dataset = dataset.map(_parser, num_parallel_calls=8)

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(4)

    return dataset


def build_test_dataset(input_tfrecord_files, batch_size):
    drop_remainder = False

    feature_description = {
        'ref_aa': tf.io.FixedLenFeature([], tf.int64),
        'alt_aa': tf.io.FixedLenFeature([], tf.int64),
        'feature': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
        'var_id': tf.io.FixedLenFeature([], tf.string),
    }

    def _parser(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        ref_aa, alt_aa = parsed['ref_aa'], parsed['alt_aa']
        var_id = parsed['var_id']

        ref_aa, alt_aa = tf.cast(ref_aa, tf.int32), tf.cast(alt_aa, tf.int32)

        feature = tf.io.decode_raw(parsed['feature'], tf.float32)
        feature = tf.reshape(feature, (window_size, input_feature_dim))

        mask = tf.io.decode_raw(parsed['mask'], tf.float32)
        mask = tf.reshape(mask, (window_size, ))
        h = window_size // 2

        #mask the postion of interest
        mask = tf.concat(
            [mask[:h],
             tf.cast([
                 1,
             ], dtype=tf.float32), mask[h + 1:]],
            axis=-1)

        return var_id, ref_aa, alt_aa, feature, mask

    dataset = tf.data.TFRecordDataset(input_tfrecord_files)

    options = tf.data.Options()
    options.experimental_threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)

    dataset = dataset.map(_parser, num_parallel_calls=8)

    dataset = dataset.batch(batch_size)

    #dataset = dataset.prefetch(4)

    return dataset
