
import copy
import math
import compress_pickle
from collections import defaultdict
import random
import scipy.stats
import statistics
import numpy as np


def calculate_hamming_distance_score(shared, nid1_connections, nid2_connections):
    n_common = 0
    for i in shared:
        if i in nid1_connections and i in nid2_connections:
            n_common += 1
        elif i not in nid1_connections and i not in nid2_connections:
            n_common += 1
    return n_common/len(shared)

def get_hamming_similarity(v0, v1):
    common = 0
    for i, j in zip(v0, v1):
        if i == j:
            common += 1
    return common / len(v0)

def calculate_z_score(shared, nid1_connections, nid2_connections):
    # construct vectors from which we can shuffle
    shared = list(shared)
    vector1 = [1 if i in nid1_connections else 0 for i in shared]
    vector2 = [1 if i in nid2_connections else 0 for i in shared]
    assert len(vector1) == len(vector2)
    observed = get_hamming_similarity(vector1, vector2)
    # compute shuffles
    shuffles = []
    for i in range(100):
        random.shuffle(vector1)
        random.shuffle(vector2)
        shuffles.append(get_hamming_similarity(vector1, vector2))
    mean = sum(shuffles)/len(shuffles)
    sd = statistics.stdev(shuffles)
    if sd == 0:
        sd = 1
    try:
        zscore = (observed-mean)/sd
    except:
        print(vector1)
        print(vector2)
        print(observed)
        print(mean)
        print(sd)
    return zscore

def calculate_convergence_score(shared, nid1_connections, nid2_connections):
    n_common = 0
    n_noncommon = 0
    for i in shared:
        if i in nid1_connections and i in nid2_connections:
            n_common += 1
        if i not in nid1_connections and i not in nid2_connections:
            n_noncommon += 1
    return n_common/(len(shared)-n_noncommon)


def calculate_pearson_correlation_score(shared, nid1_connections, nid2_connections):
    x = []
    y = []
    for i in shared:
        if i in nid1_connections:
            x.append(0 + random.random()/100)
        else:
            x.append(1 + random.random()/100)
        if i in nid2_connections:
            y.append(0 + random.random()/100)
        else:
            y.append(1 + random.random()/100)

    # # handle an error condition where an input is all constants (0 or 1)
    # x = np.asarray(x, dtype=np.float32)
    # y = np.asarray(y, dtype=np.float32)
    # # If an input is constant, the correlation coefficient is not defined.
    # if (x == x[0]).all():
    #     print(x)
    #     x[0] += .01
    #     assert not (x == x[0]).all()
    # if (y == y[0]).all():
    #     print(y)
    #     y[0] = y[0] + .01
    #     print(y)
    #     assert not (y == y[0]).all()
    return float(scipy.stats.pearsonr(x, y)[0])


def calculate_spearman_correlation_score(shared, nid1_connections, nid2_connections):
    x = []
    y = []
    for i in shared:
        if i in nid1_connections:
            x.append(0 + random.random()/100)
        else:
            x.append(1 + random.random()/100)
        if i in nid2_connections:
            y.append(0 + random.random()/100)
        else:
            y.append(1 + random.random()/100)
    return float(scipy.stats.spearmanr(x, y)[0])


class WeightDatabase():

    def __init__(
            self,
            syn_db=None,
            contact_db=None,
            ):
        self.presyns = set()
        self.postsyns = set()
        self.presyn_weights = defaultdict(lambda: defaultdict(list))
        self.postsyn_weights = defaultdict(lambda: defaultdict(list))
        self.presyn_nons = defaultdict(set)
        self.postsyn_nons = defaultdict(set)
        if syn_db:
            self.load_syn_db(syn_db)
        if contact_db:
            self.load_contact_db(contact_db)
        self.weight_fn = self.compute_weight
        self.connection_rate = dict()
        self.postsyn_connection_rate = None
        self.presyn_connection_rate = None

    def load_syn_db(self, fname, weight_fn=None):
        if weight_fn is None:
            weight_fn = self.weight_fn
        syn_db = compress_pickle.load(fname)
        for presyn_id in syn_db:
            self.presyns.add(presyn_id)
            for postsyn_id in syn_db[presyn_id]:
                self.postsyns.add(postsyn_id)
                for syn in syn_db[presyn_id][postsyn_id]:
                    weight = weight_fn(syn)
                    if weight == 0:
                        continue
                    self.presyn_weights[presyn_id][postsyn_id].append(weight)
                    self.postsyn_weights[postsyn_id][presyn_id].append(weight)

    def load_touch_db(self, fname, max_dist=200):
        touch_db = compress_pickle.load(fname)
        for presyn_id in touch_db:
            self.presyns.add(presyn_id)
            for postsyn_id in touch_db[presyn_id]:
                self.postsyns.add(postsyn_id)
                dist, _ = touch_db[presyn_id][postsyn_id]
                if dist <= max_dist:
                    self.presyn_nons[presyn_id].add(postsyn_id)
                    self.postsyn_nons[postsyn_id].add(presyn_id)

    def compute_weight(self, syn):
        # correct for prediction and compute the area of synapse
        z_len = syn['z_length'] - 40
        major_axis_length = syn['major_axis_length'] * .9
        diameter = max(z_len, major_axis_length)
        r = diameter/2
        area = math.pi*r*r
        return area

    def get_weights(self):
        return self.presyn_weights

    def get_presyn_ids(self):
        return self.presyns

    def get_postsyn_ids(self):
        return self.postsyns

    def get_connections(self, nid):
        if nid in self.presyns:
            weights = self.presyn_weights
        else:
            weights = self.postsyn_weights
        return list(weights[nid].keys())

    def get_nonconnections(self, nid):
        if nid in self.presyns:
            nons = self.presyn_nons
        else:
            nons = self.postsyn_nons
        return list(nons[nid])

    def get_total_connections(self, nid):
        if nid in self.presyns:
            weights = self.presyn_weights
            nons = self.presyn_nons
        else:
            weights = self.postsyn_weights
            nons = self.postsyn_nons
        return list(set(weights[nid].keys()) | set(nons[nid]))

    def get_shared_presyns(self, nid1, nid2):
        ret = set(self.get_total_connections(nid1))
        ret = ret & set(self.get_total_connections(nid2))
        return ret

    def calc_pattern_similarity(self, nid1, nid2):
        shared = self.get_shared_presyns(nid1, nid2)
        nid1_connections = self.get_connections(nid1)
        nid2_connections = self.get_connections(nid2)
        return calculate_hamming_distance_score(shared, nid1_connections, nid2_connections)

    def calc_pattern_convergence(self, nid1, nid2):
        shared = self.get_shared_presyns(nid1, nid2)
        nid1_connections = self.get_connections(nid1)
        nid2_connections = self.get_connections(nid2)
        return calculate_convergence_score(shared, nid1_connections, nid2_connections)

    def calc_pattern_correlation(self, nid1, nid2, spearman=False):
        shared = self.get_shared_presyns(nid1, nid2)
        nid1_connections = self.get_connections(nid1)
        nid2_connections = self.get_connections(nid2)
        if spearman:
            return calculate_spearman_correlation_score(shared, nid1_connections, nid2_connections)
        else:
            return calculate_pearson_correlation_score(shared, nid1_connections, nid2_connections)

    def calc_pattern_zscore(self, nid1, nid2):
        shared = self.get_shared_presyns(nid1, nid2)
        nid1_connections = self.get_connections(nid1)
        nid2_connections = self.get_connections(nid2)
        return calculate_z_score(shared, nid1_connections, nid2_connections)

    def calc_global_connection_rate(self, nid):
        if nid in self.postsyns:
            if self.postsyn_connection_rate is None:
                total = 0
                connected = 0
                for nid in self.postsyns:
                    total += len(self.get_total_connections(nid))
                    connected += len(self.get_connections(nid))
                self.postsyn_connection_rate = connected / total
            return self.postsyn_connection_rate
        else:
            if self.presyn_connection_rate is None:
                total = 0
                connected = 0
                for nid in self.presyns:
                    total += len(self.get_total_connections(nid))
                    connected += len(self.get_connections(nid))
                self.presyn_connection_rate = connected / total
            return self.presyn_connection_rate

    def calc_connection_rate(self, nid, global_rate=False):
        if not global_rate:
            n_total = 0
            n_connected = 0
            all_others = self.get_total_connections(nid)
            all_connected = self.get_connections(nid)
            if len(all_others) == 0:
                return 0
            return len(all_connected) / len(all_others)
        else:
            return self.calc_global_connection_rate(nid)

    def randomize_connectivity(self, type, global_rate=False):
        new_graph = copy.deepcopy(self)
        if type == 'postsyn':
            for nid in self.postsyns:
                if nid not in self.connection_rate:
                    self.connection_rate[nid] = self.calc_connection_rate(nid, global_rate)
                all_others = self.get_total_connections(nid)
                new_graph.postsyn_weights[nid] = dict()
                new_graph.postsyn_nons[nid] = set()
                rate = self.connection_rate[nid]
                for other in all_others:
                    if random.random() < rate:
                        new_graph.postsyn_weights[nid][other] = 1
                    else:
                        new_graph.postsyn_nons[nid].add(other)
        else:
            for nid in self.presyns:
                if nid not in self.connection_rate:
                    self.connection_rate[nid] = self.calc_connection_rate(nid, global_rate)
                all_others = self.get_total_connections(nid)
                new_graph.presyn_weights[nid] = dict()
                new_graph.presyn_nons[nid] = set()
                rate = self.connection_rate[nid]
                for other in all_others:
                    if random.random() < rate:
                        new_graph.presyn_weights[nid][other] = 1
                    else:
                        new_graph.presyn_nons[nid].add(other)
        return new_graph



