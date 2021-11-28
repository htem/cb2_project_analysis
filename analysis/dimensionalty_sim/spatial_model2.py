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

class GranuleCell():
    def __init__(self):
        self.id = None
        self.edges = []
        self.loc = None
        self.adjacent_boutons = None
        self.distance_cache = defaultdict(list)

    def compute_and_cache_distance(self, bouton_id, bouton_xyz):
        # print(self.loc)
        # print(bouton_xyz)
        dist = np.linalg.norm((
            (self.loc[0]-bouton_xyz[0]),
            (self.loc[1]-bouton_xyz[1]),
            (self.loc[2]-bouton_xyz[2]),
            ))
        dist = int(dist)
        self.distance_cache[dist].append(bouton_id)

class MossyFiber():
    def __init__(self):
        self.edges = []
        self.boutons = []

class Bouton():
    def __init__(self):
        self.mf_id = None
        self.edges = []

class SpatialModel():
    def __init__(
            self,
            n_grcs,
            actual_n_grcs,
            n_mfs,
            n_boutons,
            size_xyz,
            dendrite_count_dist,
            dendrite_len_dist,
            mf_size_dist,
            x_expansion,
            box_size,
            seed=0,
            ):
        self.n_grcs = n_grcs
        self.n_mfs = n_mfs
        self.n_boutons = n_boutons
        self.size_xyz = [k for k in size_xyz]
        self.actual_n_grcs = actual_n_grcs

        # z_scale = actual_n_grcs/n_grcs
        # self.size_xyz[2] *= z_scale
        # self.n_mfs *= z_scale
        # self.n_boutons *= z_scale

        self.dendrite_count_dist = dendrite_count_dist
        self.dendrite_len_dist = dendrite_len_dist
        self.mf_size_dist = mf_size_dist
        self.mfs = {}
        self.grcs = {}
        self.boutons = {}
        self.box_size = box_size
        self.x_expansion = x_expansion

        mean = statistics.mean(dendrite_count_dist)
        stdev = statistics.stdev(dendrite_count_dist, mean)
        print(f'dendrite_count_dist: mean: {mean}, std: {stdev}')
        mean = statistics.mean(dendrite_len_dist)
        stdev = statistics.stdev(dendrite_len_dist, mean)
        print(f'dendrite_len_dist: mean: {mean}, std: {stdev}')
        mean = statistics.mean(mf_size_dist)
        stdev = statistics.stdev(mf_size_dist, mean)
        print(f'mf_size_dist: mean: {mean}, std: {stdev}')

        self.expanded_size = [k for k in self.size_xyz]
        self.expanded_size[0] += 2*self.x_expansion
        self.expanded_size[2] += 2*self.x_expansion
        self.len_threshold = 4

        self.randomize(seed)

    def get_box_loc(self, loc):
        return (int(loc[0]/self.box_size),
                int(loc[1]/self.box_size),
                int(loc[2]/self.box_size))

    def get_adjacent_boutons(self, grc):
        if grc.adjacent_boutons is not None:
            return grc.adjacent_boutons

        box_loc = grc.box_loc
        adjacent_boxes = itertools.product(
                                [box_loc[0]-1, box_loc[0], box_loc[0]+1],
                                [box_loc[1]-1, box_loc[1], box_loc[1]+1],
                                [box_loc[2]-1, box_loc[2], box_loc[2]+1],
                                )

        ret = []
        for box in adjacent_boxes:
            # print(len(self.bouton_by_box[box]))
            ret.extend(self.bouton_by_box[box])
        # print(len(ret))
        grc.adjacent_boutons = ret
        # asdf
        return ret


    def init_mfs(self):
        # create mfs, mfs_id
        # assign mf_size dist
        expansion_scale = self.expanded_size[0]/self.size_xyz[0]
        expansion_scale *= self.expanded_size[2]/self.size_xyz[2]
        print(f'Making {self.n_mfs} MFs')
        for i in range(self.n_mfs):
            mf = MossyFiber()
            self.mfs[i] = mf

        # assign loc to each bouton
        # assign bouton_loc to mfs_id, add to bouton_by_box
        expanded_n_boutons = int(self.n_boutons*expansion_scale)
        print(f'Making {expanded_n_boutons} boutons')
        print(f'density={expanded_n_boutons/self.expanded_size[0]/self.expanded_size[1]/self.expanded_size[2]}')
        mf_id = 0
        size_i = 0
        for bouton_id in range(expanded_n_boutons):
            if size_i == 0:
                random.shuffle(self.mf_size_dist)
            bouton = Bouton()
            loc = (random.random()*self.expanded_size[0],
                   random.random()*self.expanded_size[1],
                   random.random()*self.expanded_size[2])
            bouton.loc = loc
            bouton.mf_id = mf_id
            bouton.size = self.mf_size_dist[size_i]
            self.mfs[mf_id].boutons.append(bouton_id)
            box_loc = self.get_box_loc(loc)
            bouton.box_loc = box_loc
            self.bouton_by_box[box_loc].append(bouton_id)
            self.boutons[bouton_id] = bouton
            mf_id += 1
            if mf_id == self.n_mfs:
                mf_id = 0
            size_i += 1
            if size_i == len(self.mf_size_dist):
                size_i = 0

    def init_grcs(self):
        z_scale = self.actual_n_grcs/self.n_grcs
        actual_xyz = [k for k in self.size_xyz]
        actual_xyz[2] *= z_scale
        x_offset = self.x_expansion
        z_offset = self.x_expansion
        print(f'actual_xyz: {actual_xyz}')

        print(f'Making {self.actual_n_grcs} grcs')
        print(f'density={self.actual_n_grcs/actual_xyz[0]/actual_xyz[1]/actual_xyz[2]}')
        dendrite_count_i = 0
        for grc_id in range(self.actual_n_grcs):
            if dendrite_count_i == 0:
                random.shuffle(self.dendrite_count_dist)
            loc = [random.random()*actual_xyz[0],
                   random.random()*actual_xyz[1],
                   random.random()*actual_xyz[2]]
            loc[0] += x_offset
            loc[2] += z_offset
            loc = tuple(loc)
            grc = GranuleCell()
            grc.id = grc_id
            grc.loc = loc
            grc.num_dendrites = self.dendrite_count_dist[dendrite_count_i]
            assert grc.num_dendrites > 0
            dendrite_count_i += 1
            if dendrite_count_i == len(self.dendrite_count_dist):
                dendrite_count_i = 0
            box_loc = self.get_box_loc(loc)
            grc.box_loc = box_loc
            self.grcs_by_box[box_loc].append(grc_id)
            self.grcs[grc_id] = grc

        print(f'Compute distances...')
        for grc_id, grc in self.grcs.items():
            boutons = self.get_adjacent_boutons(grc)
            for b in boutons:
                grc.compute_and_cache_distance(b, self.boutons[b].loc)
            # for dist in sorted(grc.distance_cache.keys()):
            #     print(f'{dist}: {grc.distance_cache[dist]}')
            #     if dist > 25:
            #         break

    def connect_wiring(self):
        '''
        shuffle grc list
        shuffle dendrite length dist
        for each grc
            get a dendrite length
            get all possible boutons
            compute and cache distances
        '''

        grc_ids = []
        for grc_id, grc in self.grcs.items():
            # print([grc_id]*grc.num_dendrites)
            grc_ids.extend([grc_id]*grc.num_dendrites)

        random.shuffle(grc_ids)

        dendrite_len_i = 0
        for grc_id in grc_ids:
            if dendrite_len_i == 0:
                random.shuffle(self.dendrite_len_dist)
            dendrite_len = self.dendrite_len_dist[dendrite_len_i]
            dendrite_len_i += 1
            if dendrite_len_i == len(self.dendrite_len_dist):
                dendrite_len_i = 0

            grc = self.grcs[grc_id]
            bouton_ids = self.get_possible_boutons(grc, dendrite_len)

            # filter out full boutons
            f_bouton_ids = []
            for bid in bouton_ids:
                if len(self.boutons[bid].edges) < self.boutons[bid].size:
                    f_bouton_ids.append(bid)
            if len(f_bouton_ids) > 0:
                bouton_ids = f_bouton_ids

            bid = bouton_ids[random.randrange(len(bouton_ids))]
            grc.edges.append(bid)
            self.boutons[bid].edges.append(grc_id)

    def prune(self):

        used_boutons = []
        bouton_sizes = []
        for bid, bouton in self.boutons.items():
            if len(bouton.edges):
                used_boutons.append((bid, bouton.loc))
                # if not (bouton.loc[0] > (40+80)
                #         and bouton.loc[0] < (120+80)):
                #     continue
                # if not (bouton.loc[2] > (50+80)
                #         and bouton.loc[2] < (70+80)):
                #     continue
                # bouton_sizes.append(len(bouton.edges))

        # print(f'num boutons: {len(bouton_sizes)}')
        # mean = statistics.mean(bouton_sizes)
        # stdev = statistics.stdev(bouton_sizes, mean)
        # print(f'Mean: {mean}, std: {stdev}')

        print(f'num boutons: {len(used_boutons)}')

        # reorder bouton ids by z
        used_boutons = sorted(used_boutons, key=lambda x: x[1][2])
        # bouton_remap_old_new = {}
        # for i, bid in enumerate(used_boutons):
        #     bouton_remap_old_new[bid] = i

        # now reorder MFs
        mf_remap_old_new = {}
        # used_boutons = sorted(used_boutons, key=lambda x: x[1][2], reverse=True)
        new_mf_id = 0
        for bid in used_boutons:
            bid = bid[0]
            mf_id = self.boutons[bid].mf_id
            assert mf_id is not None
            if mf_id not in mf_remap_old_new:
                mf_remap_old_new[mf_id] = new_mf_id
                new_mf_id += 1

        sorted_grcs = []
        for i, grc in self.grcs.items():
            sorted_grcs.append((i, grc.loc))
        sorted_grcs.sort(key=lambda x: x[1][2])

        old_grcs = self.grcs
        self.grcs = {}
        used_mfs = set()
        for new_grc_id, old_grc_id in enumerate(sorted_grcs):
            old_grc_id = old_grc_id[0]
            new_grc = GranuleCell()
            old_grc = old_grcs[old_grc_id]
            bouton_ids = old_grc.edges
            for bouton_id in bouton_ids:
                old_mf_id = self.boutons[bouton_id].mf_id
                new_mf_id = mf_remap_old_new[old_mf_id]
                used_mfs.add(new_mf_id)
                new_grc.edges.append((new_mf_id, None))
            # mf_id = int(random.random()*self.n_mfs)
            self.grcs[new_grc_id] = new_grc

        print(f'num mfs: {len(used_mfs)}')
        self.mfs = {}
        for mf_id in used_mfs:
            self.mfs[mf_id] = None


    def get_possible_boutons(self, grc, dendrite_len):
        ret = []
        for i in range(dendrite_len-self.len_threshold, dendrite_len+self.len_threshold):
            ret.extend(grc.distance_cache[i])

        if len(ret) == 0:
            len_threshold = self.len_threshold
            while len(ret) == 0:
                len_threshold = len_threshold * 2
                for i in range(dendrite_len-len_threshold, dendrite_len+len_threshold):
                    ret.extend(grc.distance_cache[i])
                print(f'expanded for {grc.id}')

        # if len(ret) == 0:
        #     print(grc.loc)
        #     print(dendrite_len)
        #     for dist in sorted(grc.distance_cache.keys()):
        #         print(f'{dist}: {grc.distance_cache[dist]}')
        #         if dist > 25:
        #             break
        assert len(ret)
        return ret

    def randomize(self, seed=0):
        if seed:
            random.seed(seed)

        self.bouton_by_box = defaultdict(lambda: [])
        self.grcs_by_box = defaultdict(lambda: [])

        self.init_mfs()
        self.init_grcs()
        self.connect_wiring()
        self.prune()

        # for i in range(self.n_boutons):
        #     loc = (random.random()*self.expanded_size[0],
        #            random.random()*self.expanded_size[1],
        #            random.random()*self.expanded_size[2])

        # self.grcs = {}
        # used_mfs = set()
        # for grc_id in range(self.n_grcs):
        #     grc = GranuleCell()
        #     mf_id = int(random.random()*self.n_mfs)
        #     grc.edges.append((mf_id, None))
        #     used_mfs.add(mf_id)
        #     self.grcs[grc_id] = grc

        # self.mfs = {}
        # for mf_id in used_mfs:
        #     self.mfs[mf_id] = None

    def remove_empty_mfs(self):
        pass
