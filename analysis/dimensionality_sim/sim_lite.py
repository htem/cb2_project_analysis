import random
import math
import copy
import numpy as np
import logging
import collections

import pyximport
# pyximport.install()
pyximport.install(setup_args={
                              # "script_args":["--compiler=mingw32"],
                              "include_dirs":np.get_include()},
                  # reload_support=True
                  )
import cython_encode


def cython_normalize(input_pattern, normalize_f, binary_sum):
    assert input_pattern.data.c_contiguous
    assert input_pattern.dtype == np.uint8
    normalize = int(normalize_f*len(input_pattern))
    cython_encode.cython_normalize(
        np.ravel(input_pattern, order='A'),
        normalize,
        len(input_pattern),
        binary_sum,
        )
    return input_pattern


class SimulationLite():

    def __init__(self, sim):
        self.dendrite_counts = []
        self.dendrite_mf_map = []
        self.thresholds = []
        for grc in sim.grcs:
            self.dendrite_counts.append(len(grc.claws))
            self.thresholds.append(grc.act_lv_scale)
            for claw in grc.claws:
                assert claw <= 65535
                self.dendrite_mf_map.append(claw)

        self.dendrite_mf_map = np.array(self.dendrite_mf_map, dtype=np.uint16)
        self.dendrite_counts = np.array(self.dendrite_counts, dtype=np.uint8)
        self.thresholds = np.array(self.thresholds, dtype=np.float32)

    def encode(self, input_pattern, out_array=None, use_cython=True,
            normalize_f=None):
        n_grcs = len(self.dendrite_counts)
        if out_array is None:
            out_array = np.empty(n_grcs, dtype=np.uint8)

        if use_cython:
            assert input_pattern.data.c_contiguous
            assert out_array.data.c_contiguous
            assert self.dendrite_mf_map.data.c_contiguous
            assert self.dendrite_counts.data.c_contiguous
            assert self.thresholds.data.c_contiguous
            assert input_pattern.dtype == np.float32
            assert out_array.dtype == np.uint8
            assert self.dendrite_mf_map.dtype == np.uint16
            assert self.dendrite_counts.dtype == np.uint8
            assert self.thresholds.dtype == np.float32
            normalize = 0
            if normalize_f is not None:
                normalize = int(normalize_f*n_grcs)
            cython_encode.cython_encode(
                np.ravel(input_pattern, order='A'),
                np.ravel(self.dendrite_counts, order='A'),
                np.ravel(self.dendrite_mf_map, order='A'),
                np.ravel(self.thresholds, order='A'),
                n_grcs,
                np.ravel(out_array, order='A'),
                normalize
                )
            return out_array

        assert normalize_f is None

        dendrite_pos = 0
        for i, dendrite_count in enumerate(self.dendrite_counts):
            s = 0.0
            for j in range(dendrite_count):
                s += input_pattern[self.dendrite_mf_map[dendrite_pos]]
                dendrite_pos += 1
            if s >= self.thresholds[i]:
                out_array[i] = 1
            else:
                out_array[i] = 0
        return out_array

