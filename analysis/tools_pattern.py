import collections
import sys
import json
import random
from jsmin import jsmin
from io import StringIO
import numpy as np

def get_eucledean_dist(a, b):
    return np.linalg.norm(
        (a[0]-b[0], a[1]-b[1], a[2]-b[2]))


def compute_hamming_distance(
        a, b,
        get_all,
        get_ones,
        no_touches=False,
        dot_product=False,
        ):
    all_a = set(get_all(a))
    ones_a = set(get_ones(a))
    all_a |= ones_a
    all_b = set(get_all(b))
    ones_b = set(get_ones(b))
    all_b |= ones_b
    common_both = all_a & all_b
    if no_touches:
        assert False, "Unimplemented"
        common_both = ones_a | ones_b
        # all_a = set(ones_a)
        # all_b = set(ones_b)
    if len(common_both) == 0:
        return None
    pattern_a = ''
    pattern_b = ''
    common_both = [k for k in common_both]
    similarity = 0
    for pc_id in common_both:
        if pc_id in ones_a:
            pattern_a += '1'
        else:
            pattern_a += '0'
        if pc_id in ones_b:
            pattern_b += '1'
        else:
            pattern_b += '0'
        if pc_id in ones_a and pc_id in ones_b:
            similarity += 1
        if pc_id not in ones_a and pc_id not in ones_b:
            if not dot_product:
                similarity += 1
    similarity = float(similarity) / len(common_both)
    summary = (similarity, common_both, pattern_a, pattern_b)
    return summary

