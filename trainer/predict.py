import tensorflow as tf
import numpy as np
import time
import glob
import logging
import argparse
import json
import sys
import os
import pandas as pd

from model_attention import ModelAttention
from constant_params import input_feature_dim, window_size
from dataset import build_test_dataset, build_all_possible_missenses_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class MultiModel(tf.keras.Model):
    def __init__(self, config, model_number):
        super(MultiModel, self).__init__()

        self.models = [
            ModelAttention(config, name=str(k)) for k in range(model_number)
        ]

    def call(self, inputs, training=False, mask=None):
        res = [tf.nn.sigmoid(m(inputs)) for m in self.models]

        return tf.reduce_mean(tf.stack(res, axis=1), axis=1)

    def predict(self, inputs, training=False, mask=None):
        return self.call(inputs, training, mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_tfrec', type=str)
    parser.add_argument('--transcript_list', type=str)
    parser.add_argument('--feature_dir', type=str)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_list', type=str)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4096)

    args = parser.parse_args()

    if args.input_tfrec is not None and args.transcript_list is not None:
        print(
            '--input_tfrec and --transript_list cannot be used at the same time'
        )
        sys.exit(-1)

    if args.input_tfrec is None and args.transcript_list is None:
        print('please provide --input_tfrec or --transcript_list')
        sys.exit(-1)

    if args.transcript_list is not None and args.feature_dir is None:
        print('please provide --feature_dir if --transcript_lis is used')
        sys.exit(-1)

    if args.model_path is None and args.model_list is None:
        print('please provide --model_path or --model_list')
        sys.exit(-1)

    if args.model_list is not None and args.model_path is not None:
        print('--model_path and --model_list cannot be used at the same time')
        sys.exit(-1)

    if args.input_tfrec is not None:
        data_files = glob.glob(args.input_tfrec)
        dataset = build_test_dataset(data_files, args.batch_size)

    if args.transcript_list is not None:
        tr_list = list(
            pd.read_csv(args.transcript_list, header=None,
                        names=['name'])['name'])
        dataset = build_all_possible_missenses_dataset(tr_list,
                                                       args.feature_dir,
                                                       args.batch_size)

    with open(args.config) as f:
        config = json.load(f)

    #to construct the model by calling a dumpy sample
    bs = args.batch_size
    x = tf.ones([bs, window_size, input_feature_dim], dtype=tf.float32)
    ref_aa = tf.ones((bs, ), dtype=tf.int32)
    alt_aa = tf.ones((bs, ), dtype=tf.int32)
    mask = tf.zeros((bs, window_size), dtype=tf.float32)

    if args.model_path is not None:
        model_paths = [args.model_path]
    else:
        model_paths = []
        with open(args.model_list) as f:
            for line in f:
                m_path = line.strip()
                if os.path.exists(m_path):
                    model_paths.append(m_path)

    models = []
    for m_path in model_paths:
        model = ModelAttention(config['model'])
        model((ref_aa, alt_aa, x), training=False, mask=mask)
        model.load_weights(m_path)
        models.append(model)

    @tf.function(input_signature=[dataset.element_spec])
    def test_step(sample):
        var, ref_aa, alt_aa, feature, padding_mask = sample

        if len(models) == 0:
            pred = models[0].predict((ref_aa, alt_aa, feature),
                                     training=False,
                                     mask=padding_mask)
        else:
            preds = [
                m.predict((ref_aa, alt_aa, feature),
                          training=False,
                          mask=padding_mask) for m in models
            ]
            pred = tf.reduce_mean(tf.stack(preds, axis=1), axis=1)

        return var, pred

    def test(test_dataset):

        all_pred, all_var = [], []

        with open(args.output, 'w') as fw:
            fw.write('var\tgMVP\n')
            for step, sample in enumerate(test_dataset):
                var, pred = test_step(sample)
                for a, b in zip(var, pred):
                    fw.write(f'{a.numpy().decode()}\t{b}\n')

    test(dataset)
