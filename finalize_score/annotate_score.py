import pandas as pd
import pickle


def main():

    gene_info = pd.read_csv('./../list/GRCh38_ensembl96_geneset.csv', sep='\t')
    gene_info_dict = {}
    for n, r in gene_info.iterrows():
        gene_info_dict[r['transcript_stable_id']] = [
            r['display_label'], r['gene_stable_id']
        ]

    score = pd.read_csv('./scores/gMVP_raw_score_Feb24.tsv', sep='\t')
    info = pd.read_csv('./scores/all_possible_missense_info.csv',
                       sep='\t',
                       dtype={'chrom': str})

    #score = pd.read_csv('./scores/sample_score.csv', sep='\t')
    #info = pd.read_csv('./scores/sample_info.csv',
    #                   sep='\t',
    #                   dtype={'chrom': str})

    info = info[info['consequence'] == 'missense_variant']

    def get_var(x):
        return '_'.join([
            x['transcript_id'],
            str(x['protein_position']), x['ref_aa'], x['alt_aa']
        ])

    info['var'] = info.apply(get_var, axis=1)

    df = pd.merge(info, score, on='var', how='inner')

    #normalized by gene
    df2 = []

    _cnt = 0

    def _get_rank(x):
        nonlocal _cnt
        _cnt += 1
        return _cnt

    for n, g in df.groupby(by='transcript_id'):
        g2 = g.sort_values(by='gMVP', axis=0, ascending=True)
        _cnt = 0
        g2['gMVP_normalized'] = g2['gMVP'].apply(_get_rank) / g2.shape[0]

        genename, gene_id = gene_info_dict.get(n, ['', ''])
        g2['gene_symbol'] = genename
        g2['gene_id'] = gene_id

        df2.append(g2)

    df = pd.concat(df2, axis=0)

    df = df.sort_values(by='gMVP', axis=0, ascending=True)
    _cnt = 0
    df['gMVP_rankscore'] = df['gMVP'].apply(_get_rank) / df.shape[0]

    df = df.sort_values(by='gMVP_normalized', axis=0, ascending=True)
    _cnt = 0
    df['gMVP_normalized_rankscore'] = df['gMVP_normalized'].apply(
        _get_rank) / df.shape[0]

    score_dict = {}
    for i, r in df.iterrows():
        score_dict[r['var']] = [
            r['gMVP'], r['gMVP_normalized'], r['gMVP_rankscore'],
            r['gMVP_normalized_rankscore']
        ]
    with open('./scores/annotated_score_dict.pickle', 'wb') as fw:
        pickle.dump(score_dict, fw)

    df = df.sort_values(by=['chrom', 'pos'], ascending=[True, True])
    df = df[[
        'chrom', 'pos', 'ref', 'alt', 'gene_symbol', 'gene_id',
        'transcript_id', 'protein_position', 'ref_aa', 'alt_aa', 'ref_codon',
        'alt_codon', 'context', 'gMVP', 'gMVP_normalized', 'gMVP_rankscore',
        'gMVP_normalized_rankscore'
    ]]

    df.to_csv('./scores/annotated_score.csv', sep='\t', index=False)


if __name__ == '__main__':
    main()
