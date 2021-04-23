import os, sys
import json
from collections import defaultdict
import numpy as np
import pandas as pd

dna_pair = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
amino_acid_index_table = {
    'A': 0,
    'B': 20,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'J': 20,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'O': 20,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'U': 20,
    'V': 17,
    'W': 18,
    'X': 20,
    'Y': 19,
    'Z': 20,
}

ss3_encode = {'H': 0, 'E': 1, 'C': 2}
ss8_encode = {'H': 0, 'G': 1, 'I': 2, 'B': 3, 'E': 4, 'S': 5, 'T': 6, 'C': 7}

ss8_ss3_encode = {
    'H': 0,
    'G': 0,
    'I': 0,
    'B': 1,
    'E': 1,
    'S': 2,
    'T': 2,
    'C': 2
}


def read_fasta(fasta_path, name_rule=None):
    fasta = defaultdict(str)
    seq = ""
    print(fasta_path)
    with open(fasta_path) as f:
        for line in f:
            print(line)
            if len(line.strip()) == 0:
                continue
            if line.startswith('>'):
                print(line)
                if seq != '':
                    fasta[head] = seq
                    seq = ''
                if name_rule is None:
                    head = line.strip().split()[0][1:]
                else:
                    head = name_rule(line)
            else:
                seq += line.strip()
        if seq != "":
            fasta[head] = seq
    return fasta


def aa_index(aa):
    return amino_acid_index_table.get(aa, 20)


def get_complement_dna(dna):
    return ''.join([dna_pair.get(a, '-') for a in dna.strip()])


#reverse strand
def get_reverse_dna(dna):
    r = ''.join([dna_pair[a] for a in dna])
    return r[::-1]


def read_vep(input_path, read_id=True):
    head = [
        'grch38_chrom', 'gch38_pos', 'ref', 'alt', 'ref_codon', 'alt_codon',
        'frame', 'transcript_stable_id', 'protein_len', 'aa_pos', 'ref_aa',
        'alt_aa'
    ]

    skiprows = 0
    with open(input_path) as f:
        for line in f:
            if not line.startswith('## '):
                break
            skiprows += 1
    df = pd.read_csv(input_path, sep='\t', skiprows=skiprows)

    #filters
    df = df[df['CANONICAL'] == 'YES']
    if 'VARIANT_CLASS' in df.columns:
        df = df[df['VARIANT_CLASS'] == 'SNV']
    df = df[df['Consequence'].apply(lambda x: 'missense_variant' in x)]

    if read_id:
        df['label'] = df['#Uploaded_variation'].apply(
            lambda x: int(x.split('|')[-1]))
        df['source'] = df['#Uploaded_variation'].apply(
            lambda x: x.split('|')[2])
    else:
        df['label'] = -1
        df['source'] = 'unknown'

    df['transcript_stable_id'] = df['Feature'].apply(lambda x: x.split('.')[0])
    df = df[df['transcript_stable_id'].apply(lambda x: x in used_tr)]

    df['protein_var'] = df.apply(_get_protein_var, axis=1)
    df['var'] = df.apply(_get_var, axis=1)

    def _get_af(x):
        if type(x) == str and x == '-':
            return 0.0
        return float(x)

    df['af'] = df['gnomAD_AF'].apply(_get_af)

    def _get_frame(x):
        r = 0
        for a in x:
            if a.isupper():
                return r
            r += 1
        assert 1 == 2
        return 0

    df['frame'] = df['Codons'].apply(_get_frame)

    df = df.drop_duplicates(['var'])

    df = df[head]
    return df


def parse_uniprot_isoform():
    pass
