import random
import pandas as pd
import argparse
import traceback
import numpy as np
import os, sys
from multiprocessing import Process
import pickle

import tensorflow as tf

import util

random.seed(2020)

#shannon
feature_dir = '/data/hz2529/zion/MVPContext/combined_feature_2021_v2'
#md22
feature_dir = '/mnt/BigData/hz2529/gMVP/feature/combined_feature_2021_v2/'


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _build_one_var(r, feature, width):
    w = width
    L = feature.shape[0]
    aa_pos, ref_aa, alt_aa, label, var_id = r['aa_pos'], util.aa_index(
        r['ref_aa']), util.aa_index(r['alt_aa']), r['target'], r['var']

    aa_pos -= 1
    assert (ref_aa == feature[aa_pos, 0])
    start = max(aa_pos - w, 0)
    end = min(L, aa_pos + 1 + w)
    var_start = start - (aa_pos - w)
    var_end = var_start + (end - start)
    var_feature = np.zeros([w * 2 + 1, feature.shape[1]])
    var_feature[var_start:var_end] = feature[start:end]

    mask = np.ones((w * 2 + 1, ), dtype=np.float32)
    mask[var_start:var_end] = 0.0

    tf_feature = {}
    tf_feature['label'] = _int64_feature(int(label))
    tf_feature['ref_aa'] = _int64_feature(ref_aa)
    tf_feature['alt_aa'] = _int64_feature(alt_aa)
    tf_feature['feature'] = _bytes_feature(
        var_feature.astype(np.float32).tobytes())
    tf_feature['mask'] = _bytes_feature(mask.astype(np.float32).tobytes())

    tf_feature['var_id'] = _bytes_feature(var_id.encode('utf-8'))

    example = tf.train.Example(features=tf.train.Features(feature=tf_feature))

    return example


def build_one_transcript(df, tf_writer):
    transcript_id = df.iloc[0]['transcript_id']

    feature_path = f'{feature_dir}/{transcript_id}.pickle'

    if not os.path.exists(feature_path):
        return 0, 0

    with open(feature_path, 'rb') as fr:
        feature = pickle.load(fr)

    pos_num, neg_num = 0, 0

    for idx, r in df.iterrows():
        try:
            example = _build_one_var(r, feature, args.width)

            if example is not None:
                tf_writer.write(example.SerializeToString())
            if r['target'] == 0:
                neg_num += 1
            else:
                pos_num += 1
        except:
            traceback.print_exc()
            print(r)

    return pos_num, neg_num


def build_one_thread(curated, output):
    pos_num = 0
    neg_num = 0
    with tf.io.TFRecordWriter(f'{output}.tfrec') as writer:
        for transcript_name, df in curated.groupby('transcript_id'):
            try:
                p, n = build_one_transcript(df, writer)
                pos_num += p
                neg_num += n
            except:
                traceback.print_exc()

    print(f'thread pos_num= {pos_num} neg_num= {neg_num}')


def build(path, output, cpu, af):
    df = pd.read_csv(path, sep='\t')

    if 'consequence' in df.columns:
        df = df[df['consequence'] == 'missense_variant']
    df = df[df['target'].isin([1, 0])]

    print('target')
    print(df['target'].value_counts())
    print('transcript num')
    print(df['transcript_id'].nunique())

    df = df.sample(frac=1, random_state=2020).reset_index(drop=True)

    if cpu <= 1:
        build_one_thread(df, output)
    else:
        num_each = int((df.shape[0] - 1) / cpu) + 1
        pool = []
        for idx in range(cpu):
            start = idx * num_each
            end = start + num_each
            if idx == cpu - 1:
                end = df.shape[0]
            p = Process(target=build_one_thread,
                        args=(df.iloc[start:end], f'{output}_{idx}'))
            pool.append(p)
        for p in pool:
            p.start()
        for p in pool:
            p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--af', type=float, default=1.0)
    parser.add_argument('--width', type=int, default=64)
    args = parser.parse_args()

    build(args.input, args.output, args.cpu, args.af)
