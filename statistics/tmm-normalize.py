#!/usr/bin/env python3
#
# Copyright (c) 2014 Hyeshik Chang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

import pandas as pd
import numpy as np

def calculate_tmm_norm_factor(ref, sample, trim_m=.3, trim_a=.05):
    if np.abs(ref - sample).sum() < 1e-10:
        return 1.

    zero_positions = ((ref == 0) | (sample == 0))

    ref_nonzero = ref[~zero_positions]
    sample_nonzero = sample[~zero_positions]
    log_ref_nonzero = np.log2(ref_nonzero)
    log_sample_nonzero = np.log2(sample_nonzero)

    M = log_sample_nonzero - log_ref_nonzero
    A = (log_sample_nonzero + log_ref_nonzero) / 2

    readsum_ref = ref_nonzero.sum()
    readsum_sample = sample_nonzero.sum()
    weights = 1. / ((readsum_ref - ref_nonzero) / (readsum_ref * ref_nonzero) +
                    (readsum_sample - sample_nonzero) / (readsum_sample * sample_nonzero))

    M_trim_min, M_trim_max = M.quantile([trim_m, 1 - trim_m])
    A_trim_min, A_trim_max = A.quantile([trim_a, 1 - trim_a])

    trimming_mask = ((M > M_trim_min) & (M < M_trim_max) &
                     (A > A_trim_min) & (A < A_trim_max))
    M_trimmed = M[trimming_mask]
    weights_trimmed = weights[trimming_mask]

    return np.exp2((M_trimmed * weights_trimmed).sum() / weights_trimmed.sum())


def tmm_normalize(readcounts, trim_m=.3, trim_a=.05, reads_cutoff=100,
                  minimum_expressed_samples=2):
    expressed_genes = (readcounts >= reads_cutoff).sum(axis=1) >= minimum_expressed_samples
    expressed = readcounts[expressed_genes]

    q75_expr = expressed.apply(lambda x: x.quantile(.75))
    refsample = np.argmin(np.abs(q75_expr - q75_expr.mean()))

    normf = expressed.apply(lambda x: calculate_tmm_norm_factor(expressed[refsample], x))

    return normf / np.exp(np.log(normf).mean())


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description='Calculate normalization factors by the TMM ('
                        'trimmed mean of M values) method. Divide factors to read counts.')
    parser.add_argument('--input', dest='input', metavar='FILE', type=str, required=True,
                        help='Path to a readcount table')
    parser.add_argument('--logratio-trim', dest='logratio_trim', metavar='FRACTION',
                        type=float, default=.3,
                        help='Fraction to trim from both extremes of M values (default: .3)')
    parser.add_argument('--average-trim', dest='average_trim', metavar='FRACTION',
                        type=float, default=.05,
                        help='Fraction to trim from both extremes of A values (default: .05)')
    parser.add_argument('--read-count-cutoff', dest='min_reads', metavar='INTEGER',
                        type=int, default=100,
                        help='Minimum read count for choosing expressed genes (default: 100)')
    parser.add_argument('--num-samples-for-read-count', dest='samples_reads_cutoff',
                        metavar='INTEGER', type=int, default=2,
                        help='Minimum number of samples that must contain enough reads for '
                             'genes to be included in factors calculations (default: 2)')
    parser.add_argument('--output-normalized', dest='output_normalized',
                        metavar='FILE', type=str, default=None,
                        help='Output file for normalized read count table.')
    parser.add_argument('--output-normalization-factors', dest='output_factors',
                        metavar='FILE', type=str, default=None,
                        help='Output file for normalized read count table.')

    options = parser.parse_args()

    return options


if __name__ == '__main__':
    options = parse_arguments()

    tbl = pd.read_table(options.input, index_col=0)
    normfactors = tmm_normalize(tbl, trim_m=options.logratio_trim, trim_a=options.average_trim,
                                reads_cutoff=options.min_reads,
                                minimum_expressed_samples=options.samples_reads_cutoff)

    if options.output_normalized is not None:
        np.round(tbl / normfactors, 3).to_csv(options.output_normalized, sep='\t')

    if options.output_factors is not None:
        normfactors.to_csv(options.output_factors, header=False, sep='\t')

