#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from multiprocessing.dummy import Pool

from attrdict import AttrDict

from util.text import levenshtein


def pmap(fun, iterable):
    pool = Pool()
    results = pool.map(fun, iterable)
    pool.close()
    return results


def wer_cer_batch(samples):
    r"""
    The WER is defined as the edit/Levenshtein distance on word level divided by
    the amount of words in the original text.
    In case of the original having more words (N) than the result and both
    being totally different (all N words resulting in 1 edit operation each),
    the WER will always be 1 (N / N = 1).
    """
    wer = sum(s.word_distance for s in samples) / sum(s.word_length for s in samples)
    cer = sum(s.char_distance for s in samples) / sum(s.char_length for s in samples)

    wer = min(wer, 1.0)
    cer = min(cer, 1.0)

    return wer, cer


def process_decode_result(item):
    ground_truth, prediction, loss = item
    char_distance = levenshtein(ground_truth, prediction)
    char_length = len(ground_truth)
    word_distance = levenshtein(ground_truth.split(), prediction.split())
    word_length = len(ground_truth.split())
    return AttrDict({
        'src': ground_truth,
        'res': prediction,
        'loss': loss,
        'char_distance': char_distance,
        'char_length': char_length,
        'word_distance': word_distance,
        'word_length': word_length,
        'cer': char_distance / char_length,
        'wer': word_distance / word_length,
    })


def calculate_report(labels, decodings, losses):
    r'''
    This routine will calculate a WER report.
    It'll compute the `mean` WER and create ``Sample`` objects of the ``report_count`` top lowest
    loss items from the provided WER results tuple (only items with WER!=0 and ordered by their WER).
    '''
    samples = pmap(process_decode_result, zip(labels, decodings, losses))

    # Getting the WER and CER from the accumulated edit distances and lengths
    samples_wer, samples_cer = wer_cer_batch(samples)

    # Order the remaining items by their loss (lowest loss on top)
    samples.sort(key=lambda s: s.loss)

    # Then order by WER (highest WER on top)
    samples.sort(key=lambda s: s.wer, reverse=True)

    return samples_wer, samples_cer, samples
