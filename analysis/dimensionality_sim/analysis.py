# from bitarray import bitarray
# import random
import math
import statistics
# import copy
import numpy as np
# import logging
import collections
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

# def get_average_hamming_distance(acts, ref_output):
#     dist_sum = 0
#     ref_output = [neuron_out[0] for neuron_out in ref_output]
#     num_samples = len(acts[0])
#     pattern_len = len(ref_output)
#     for i in range(num_samples):
#         pattern = [neuron_out[i] for neuron_out in acts]
#         for j in range(pattern_len):
#             if ref_output[j] != pattern[j]:
#                 dist_sum += 1
#     return dist_sum/num_samples

def get_average_hamming_distance2(acts, ref_output):
    dist_sum = 0
    assert len(acts[0]) == len(ref_output)
    num_samples = len(acts)
    pattern_len = len(ref_output)
    for grc_pattern in acts:
        for i, j in zip(grc_pattern, ref_output):
            if i != j:
                dist_sum += 1
    return dist_sum/num_samples

def get_hamming_distance_hist(acts, ref_output):
    hist = collections.defaultdict(int)
    for grc_pattern in acts:
        for n, ij in enumerate(zip(grc_pattern, ref_output)):
            i, j = ij
            if i != j:
                hist[n] += 1

    hist = [(k, v) for k, v in hist.items()]
    hist.sort(key=lambda x: x[1], reverse=True)
    hist = [x[1] for x in hist]
    # print(hist[0:40])
    # print(hist)
    # print(hist[100:140])
    return hist


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

# def get_average_metric(acts, ref_output, metric):
#     # ref_output = [neuron_out[0] for neuron_out in ref_output]
#     num_samples = len(acts[0])
#     total = 0
#     for i in range(num_samples):
#         pattern = [neuron_out[i] for neuron_out in acts]
#         if metric == 'voi':
#             total += get_binary_voi(ref_output, pattern)
#         elif metric == 'binary_similarity':
#             total += get_binary_similarity(ref_output, pattern)
#     return total/num_samples


def get_optimal_weights_change(act0, act1,
        valence_dir='01',
        irrelevant_bits='0',
        seed=0):
    weights = []
    assert len(act0) == len(act1)
    for a0, a1 in zip(act0, act1):
        if a0 < a1:
            weights.append(1 if valence_dir == '01' else 0)
        elif a0 > a1:
            weights.append(1 if valence_dir == '10' else 0)
        else:
            if irrelevant_bits == '0':
                weights.append(0)
            elif irrelevant_bits == '1':
                weights.append(1)
            elif irrelevant_bits == 'random':
                weights.append(random.randint(0, 1))
            elif irrelevant_bits == 'plus':
                # set weight where there is potential for even more difference in the valence_dir
                if valence_dir == '01':
                    weights.append(1 if a0 == 0 else 0)
                elif valence_dir == '10':
                    weights.append(1 if a0 == 1 else 0)
                else: assert 0
            else: assert 0

    assert len(act0) == len(weights)
    return weights


def get_directional_distance(a, b, valence_dir='01'):
    weights = get_optimal_weights_change(a, b, irrelevant_bits='0', valence_dir=valence_dir)
    return sum(weights)


def get_output_deviation(acts):
    sums = []
    for act in acts:
        sums.append(sum(act))
    mean = statistics.mean(sums)
    return mean, statistics.stdev(sums, mean)

def get_average_metric2(acts, ref_output, metric):
    # ref_output = [neuron_out[0] for neuron_out in ref_output]
    num_samples = len(acts)
    total = 0
    for pattern in acts:
        if metric == 'voi':
            total += get_binary_voi(ref_output, pattern)
        elif metric == 'binary_similarity':
            total += get_binary_similarity(ref_output, pattern)
        elif metric == 'dir_distance_01':
            total += get_directional_distance(ref_output, pattern, valence_dir='01')
        elif metric == 'dir_distance_10':
            total += get_directional_distance(ref_output, pattern, valence_dir='10')
    return total/num_samples

