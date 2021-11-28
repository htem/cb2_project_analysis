
cimport cython
cimport numpy as np
from libc.stdlib cimport rand

def cython_encode(
    np.ndarray[np.float32_t, ndim=1, mode='c'] input_pattern,
    np.ndarray[np.uint8_t, ndim=1, mode='c'] dendrite_count_array,
    np.ndarray[np.uint16_t, ndim=1, mode='c'] dendrite_mf_map_array,
    np.ndarray[np.float32_t, ndim=1, mode='c'] threshold_array,
    int n_grcs,
    np.ndarray[np.uint8_t, ndim=1, mode='c'] out_array,
    int normalize,
    ):

    cdef int i, j
    cdef int dendrite_count, dendrite_pos, mf_id, 
    cdef float s
    i = 0
    dendrite_pos = 0
    while i < n_grcs:
        j = 0
        s = 0
        dendrite_count = dendrite_count_array[i]
        while j < dendrite_count:
            mf_id = dendrite_mf_map_array[dendrite_pos]
            dendrite_pos += 1
            s += input_pattern[mf_id]
            j += 1
        if s >= threshold_array[i]:
            out_array[i] = 1
        else:
            out_array[i] = 0
        i += 1

    cdef int binary_sum, diff, random_i
    if normalize > 0:
        i = 0
        binary_sum = 0
        while i < n_grcs:
            binary_sum += out_array[i]
            i += 1
        diff = normalize - binary_sum
        # print(diff)
        while diff != 0:
            random_i = rand() % n_grcs
            if diff < 0:
                if out_array[random_i] == 1:
                    out_array[random_i] = 0
                    diff += 1
            elif diff > 0:
                if out_array[random_i] == 0:
                    out_array[random_i] = 1
                    diff -= 1


def cython_normalize(
    np.ndarray[np.uint8_t, ndim=1, mode='c'] out_array,
    int normalize,
    int length,
    int binary_sum,
    ):

    # cdef int binary_sum, diff, random_i
    cdef int diff, random_i
    i = 0
    # binary_sum = 0
    # while i < length:
    #     binary_sum += out_array[i]
    #     i += 1
    # print(binary_sum)
    # print(length)
    diff = normalize - binary_sum
    # print(diff)
    while diff != 0:
        random_i = rand() % length
        if diff < 0:
            if out_array[random_i] == 1:
                out_array[random_i] = 0
                diff += 1
        elif diff > 0:
            if out_array[random_i] == 0:
                out_array[random_i] = 1
                diff -= 1



