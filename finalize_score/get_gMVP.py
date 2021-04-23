import argparse
import os
import numpy as np

import pickle
import pandas as pd

score_path = '/home/hz2529/repos/gMVP/finalize_score/scores/annotated_score_dict.pickle'


def load_all_score():
    with open(score_path, 'rb') as fr:
        scores = pickle.load(fr)

    return scores


def get_score(scores, in_file, output_file):
    df = pd.read_csv(in_file, sep='\t')
    df['gMVP'], df['gMVP_normalized'], df['gMVP_rankscore'] = zip(
        *df['protein_var'].apply(
            lambda x: scores.get(x, [np.nan, np.nan, np.nan, np.nan])[:3]))

    df.to_csv(output_file, sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        type=str,
                        action='append',
                        nargs='+',
                        required=True)
    args = parser.parse_args()

    in_files = []
    for f in args.input:
        for ff in f:
            if os.path.exists(ff):
                in_files.append(ff)

    if len(in_files) > 0:
        scores = load_all_score()

    for in_file in in_files:
        output_file = '.'.join(in_file.split('.')[:-1] + ['gMVP.csv'])
        print(in_file, output_file)
        get_score(scores, in_file, output_file)
