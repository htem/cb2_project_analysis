# from bitarray import bitarray
import random
# import math
import copy
import numpy as np
# import logging
import itertools
from collections import defaultdict
import statistics


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

# class MossyFiber():
#     def __init__(self):
#         self.edges = []
#         self.boutons = []

# class Bouton():
#     def __init__(self):
#         self.mf_id = None
#         self.edges = []

class ScaleUpModel():
    def __init__(
            self,
            input_graph,
            n_grcs,
            # actual_n_grcs,
            n_mfs,
            # n_boutons,
            # size_xyz,
            # dendrite_count_dist,
            # dendrite_len_dist,
            # mf_size_dist,
            # x_expansion,
            # box_size,
            seed=0,
            sort_mfs_by_z=False
            ):
        self.n_grcs = n_grcs
        self.n_mfs = n_mfs
        self.input_graph = input_graph
        self.sort_mfs_by_z = sort_mfs_by_z

        self.mfs = {}
        self.grcs = {}

        print(f'input n_grcs: {len(input_graph.grcs)}')
        print(f'input n_nfs: {len(input_graph.mfs)}')

        # mean = statistics.mean(dendrite_count_dist)
        # stdev = statistics.stdev(dendrite_count_dist, mean)
        # print(f'dendrite_count_dist: mean: {mean}, std: {stdev}')
        # mean = statistics.mean(dendrite_len_dist)
        # stdev = statistics.stdev(dendrite_len_dist, mean)
        # print(f'dendrite_len_dist: mean: {mean}, std: {stdev}')
        # mean = statistics.mean(mf_size_dist)
        # stdev = statistics.stdev(mf_size_dist, mean)
        # print(f'mf_size_dist: mean: {mean}, std: {stdev}')

        self.randomize(seed)


    def simplify_input_graph(self, input_graph):
        simplifed_grcs = []
        mapping = {}
        counter = 0
        used_mfs = set()
        for grc_id, grc in input_graph.grcs.items():
            for e, _ in grc.edges:
                used_mfs.add(e)
        if not self.sort_mfs_by_z:
            # asdf
            for mf_id, mf in input_graph.mfs.items():
                if mf_id not in used_mfs:
                    continue
                mapping[mf_id] = counter
                counter += 1
        else:
            ordering = []
            for mf_id, mf in input_graph.mfs.items():
                if mf_id not in used_mfs:
                    continue
                for loc in mf.locs:
                    ordering.append((mf_id, loc[2]))
            ordering.sort(key=lambda x: x[1])
            # print(ordering)
            processed = set()
            for mf_id, _ in ordering:
                if mf_id in processed:
                    continue
                processed.add(mf_id)
                mapping[mf_id] = counter
                counter += 1

        for grc_id, grc in input_graph.grcs.items():
            claws = [mapping[mf_id] for mf_id, _ in grc.edges]
            grc = GranuleCell()
            grc.edges = claws
            simplifed_grcs.append(grc)
        return simplifed_grcs, counter

    def randomize(self, seed=0):
        random.seed(seed)
        input_grcs, input_n_mfs = self.simplify_input_graph(self.input_graph)
        print(f'true input n_nfs: {input_n_mfs}')

        # replicate
        prededup_grcs = []
        prededup_n_mfs = 0
        while len(prededup_grcs) < self.n_grcs:
            for input_grc in input_grcs:
                new_grc = GranuleCell()
                for edge in input_grc.edges:
                    new_grc.edges.append(edge+prededup_n_mfs)
                prededup_grcs.append(new_grc)
            prededup_n_mfs += input_n_mfs

        print(f'prededup_n_grcs: {len(prededup_grcs)}')
        print(f'prededup_n_mfs: {prededup_n_mfs}')

        # drop extra grcs
        to_ignore_grcs = set()
        while (len(prededup_grcs) - len(to_ignore_grcs)) > self.n_grcs:
            to_ignore_grcs.add(random.randrange(len(prededup_grcs)))
        print(f'to_ignore_grcs: {len(to_ignore_grcs)}')

        # assigning mfs
        mf_remap = {}
        mfs_shuffler = Shuffler([i for i in range(self.n_mfs)])
        for old_mf_id in range(prededup_n_mfs):
            new_mf_id = mfs_shuffler.get_one()
            mf_remap[old_mf_id] = new_mf_id

        # wrap up
        self.grcs = {}
        used_mfs = set()
        grc_id = 0
        for old_grc_id in range(len(prededup_grcs)):
            if old_grc_id in to_ignore_grcs:
                continue
            new_grc = GranuleCell()
            old_grc = prededup_grcs[old_grc_id]
            for edge in old_grc.edges:
                new_mf_id = mf_remap[edge]
                new_grc.edges.append((new_mf_id, None))
                used_mfs.add(new_mf_id)
            self.grcs[grc_id] = new_grc
            grc_id += 1
        assert len(self.grcs) == self.n_grcs

        print(f'used_mfs: {len(used_mfs)}')
        self.mfs = {}
        for mf_id in used_mfs:
            self.mfs[mf_id] = None
        assert len(self.mfs) <= self.n_mfs

    def remove_empty_mfs(self):
        return []
