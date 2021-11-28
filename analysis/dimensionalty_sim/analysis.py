# from bitarray import bitarray
# import random
import math
# import copy
import numpy as np
# import logging
# import collections
from numpy import linalg as LA

def get_covariance_matrix(data):
    arr = np.array(data)
    return np.cov(arr, bias=False)
    # return np.cov(arr, bias=True)
    # covMatrix = np.cov(data, bias=True)

def get_eigenvalues(arr):
    w, _ = LA.eig(arr)
    return np.real(w)

def get_dim_from_eigenvalues(evs):
    square_of_sum = 0
    sum_of_square = 0
    for ev in evs:
        square_of_sum += ev
        sum_of_square += ev*ev
    square_of_sum = square_of_sum*square_of_sum
    return square_of_sum/sum_of_square

def get_population_correlation_from_eigenvalues(evs):
    n = len(evs)
    sqrt_evs = [math.sqrt(abs(e)) for e in evs]
    max_ev = max(sqrt_evs)
    sum_of_root = 0
    for ev in sqrt_evs:
        # print(ev)
        sum_of_root += ev
    max_ev /= sum_of_root
    max_ev -= 1/n
    max_ev *= n/(n-1)
    return max_ev

def get_dim_from_acts(acts, ret_population_correlation=False):
    cov = get_covariance_matrix(acts)
    ev = get_eigenvalues(cov)
    dim = get_dim_from_eigenvalues(ev)
    if ret_population_correlation:
        pop_corr = get_population_correlation_from_eigenvalues(ev)
        return dim, pop_corr
    return dim

def get_average_hamming_distance(acts, ref_output):
    dist_sum = 0
    ref_output = [neuron_out[0] for neuron_out in ref_output]
    num_samples = len(acts[0])
    pattern_len = len(ref_output)
    for i in range(num_samples):
        pattern = [neuron_out[i] for neuron_out in acts]
        for j in range(pattern_len):
            if ref_output[j] != pattern[j]:
                dist_sum += 1
    return dist_sum/num_samples

def get_average_hamming_distance2(acts, ref_output):
    dist_sum = 0
    # ref_output = [neuron_out[0] for neuron_out in ref_output]
    num_samples = len(acts[0])
    pattern_len = len(ref_output)
    for pattern in acts:
        # pattern = [neuron_out[i] for neuron_out in acts]
        for i, j in zip(pattern, ref_output):
            if i != j:
                dist_sum += 1
        # for j in range(pattern_len):
            # if ref_output[j] != pattern[j]:
                # dist_sum += 1
    return dist_sum/num_samples

def get_normalized_mean_squared_distance(norm_hamming, f):
    # return norm_hamming / (2*f*(1-f))
    return norm_hamming/(2*f)/(1-f)
    # return norm_hamming/(2*f)

def get_binary_similarity(a, b):
    # print(a)
    # print(b)
    same = 0
    total = 0
    for j in range(len(a)):
        if a[j]:
            total += 1
            if b[j]:
                same += 1
    if total > 0:
        similarity = same/total
        # print(similarity)
        return similarity
    else:
        return 1

def variation_of_information(X, Y):
    # https://gist.github.com/jwcarr/626cbc80e0006b526688
    # print(X)
    # print(Y)
    n = float(sum([len(x) for x in X]))
    sigma = 0.0
    for x in X:
        p = len(x) / n
        for y in Y:
            q = len(y) / n
            r = len(set(x) & set(y)) / n
            if r > 0.0:
                sigma += r * (math.log(r / p, 2) + math.log(r / q, 2))
    return abs(sigma)

def get_binary_voi(a, b):
    assignments = []
    for v in [a, b]:
        assignment = [[], []]
        for i, e in enumerate(v):
            if e:
                assignment[0].append(i)
            else:
                assignment[1].append(i)
        assignments.append(assignment)
    return variation_of_information(assignments[0], assignments[1])

def get_average_metric(acts, ref_output, metric):
    # ref_output = [neuron_out[0] for neuron_out in ref_output]
    num_samples = len(acts[0])
    total = 0
    for i in range(num_samples):
        pattern = [neuron_out[i] for neuron_out in acts]
        if metric == 'voi':
            total += get_binary_voi(ref_output, pattern)
        elif metric == 'binary_similarity':
            total += get_binary_similarity(ref_output, pattern)
    return total/num_samples

def get_average_metric2(acts, ref_output, metric):
    # ref_output = [neuron_out[0] for neuron_out in ref_output]
    num_samples = len(acts)
    total = 0
    for pattern in acts:
        # pattern = [neuron_out[i] for neuron_out in acts]
        if metric == 'voi':
            total += get_binary_voi(ref_output, pattern)
        elif metric == 'binary_similarity':
            total += get_binary_similarity(ref_output, pattern)
    return total/num_samples
