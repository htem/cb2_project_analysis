# from bitarray import bitarray
import random
# import math
import copy
# import numpy as np
# import logging
# import collections
# logger = logging.getLogger(__name__)

class Shuffler():
    def __init__(self, dist):
        self.dist = copy.copy(dist)
        random.shuffle(self.dist)
        self.i = 0
    def get_one(self):
        ret = self.dist[self.i]
        self.i += 1
        if self.i == len(self.dist):
            self.i = 0
            random.shuffle(self.dist)
        return ret

class GranuleCell():
    def __init__(self):
        self.edges = []

class MossyFiber():
    def __init__(self):
        self.locs = None

class GlobalRandomModel():
    def __init__(
            self,
            n_grcs,
            n_mfs,
            n_dendrites=None,
            seed=None,
            ):
        self.n_grcs = n_grcs
        self.n_mfs = n_mfs
        if n_dendrites is None:
            n_dendrites = [4, 4, 5, 5]
        if seed is not None:
            random.seed(seed)
        self.n_dendrites = Shuffler(n_dendrites)
        # self.n_mfs_actual = n_mfs_actual
        self.mf_locs = []
        self.randomize(seed)

    def randomize(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.grcs = {}
        used_mfs = set()
        for grc_id in range(self.n_grcs):
            grc = GranuleCell()
            for i in range(self.n_dendrites.get_one()):
                mf_id = int(random.random()*self.n_mfs)
                grc.edges.append((mf_id, None))
                used_mfs.add(mf_id)
            self.grcs[grc_id] = grc

        print(f'n_grcs: {len(self.grcs)}')
        self.mfs = {}
        for mf_id in used_mfs:
            self.mfs[mf_id] = MossyFiber()
        print(f'n_mfs: {len(self.mfs)}')

    def make_redundant(self, redundant_factor, n_dendrite_share, seed=None):
        assert redundant_factor >= 1
        if redundant_factor == 1:
            return self
        if seed is not None:
            random.seed(seed)

        n_grcs_to_keep = int(self.n_grcs / redundant_factor +.5)
        grcs = list(self.grcs.keys())
        random.shuffle(grcs)

        new_grcs = {}
        new_grc_ids = []
        for i in range(n_grcs_to_keep):
            old_grc_id = grcs[i]
            new_grcs[i] = self.grcs[old_grc_id]
            new_grc_ids.append(i)

        print(f'Keeping {len(new_grc_ids)} grcs')

        current_grc_ids = Shuffler(new_grc_ids)
        n_grcs_to_add = self.n_grcs - n_grcs_to_keep
        for i in range(n_grcs_to_add):
            current_grc_id = current_grc_ids.get_one()
            current = new_grcs[current_grc_id]
            redundant_grc = copy.deepcopy(current)
            # remove n dendrites and add n random dendrites
            edges = redundant_grc.edges
            n_dendrite_share = min(n_dendrite_share, len(edges))
            # assert n_dendrite_share <= len(edges)
            n_remove = len(edges) - n_dendrite_share
            random.shuffle(edges)
            for j in range(n_remove):
                edges.pop()
            for j in range(n_remove):
                mf_id = int(random.random()*self.n_mfs)
                edges.append((mf_id, None))
            new_id = n_grcs_to_keep + i
            assert new_id not in new_grcs
            new_grcs[new_id] = redundant_grc

        assert len(new_grcs) == len(self.grcs)
        self.grcs = new_grcs
        return self

    def remove_empty_mfs(self):
        return []
