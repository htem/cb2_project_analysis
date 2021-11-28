
from collections import defaultdict
import random

n_mfs = 7000
n_grcs = 204000

connections = defaultdict(set)

for grc_id in range(n_grcs):
    for k in range(4):
        mf_id = int(random.random()*n_mfs)
        connections[grc_id].add(mf_id)


# input_graph.finalize_gt()

n_common_pairs = 0
total_n_pairs = 0
hist = defaultdict(int)
n = 0
for grc_i_id in range(n_grcs):
    n += 1
    if n % 100 == 0:
        print(n, end=', ', flush=True)
    if n == 2000:
        break
    rosettes_i = connections[grc_i_id]
    for grc_j_id in range(grc_i_id+1, n_grcs):
        n_common = len(rosettes_i & connections[grc_j_id])
        hist[n_common] += 1

print(hist)

import compress_pickle
compress_pickle.dump((
    n,
    hist,
    ), f"gen_global_random_7k_204k_data_{n}.gz")
# normalize

