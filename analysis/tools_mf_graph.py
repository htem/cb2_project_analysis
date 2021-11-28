import collections
import sys
import json
import random
from jsmin import jsmin
from io import StringIO
import numpy as np
from collections import defaultdict
import copy

def to_ng_coord(coord):
    return (
        int(coord[0]/4),
        int(coord[1]/4),
        int(coord[2]/40),
        )

def get_eucledean_dist(a, b):
    return np.linalg.norm(
        (a[0]-b[0], a[1]-b[1], a[2]-b[2]))

def get_distance(u, v):
    return int(get_eucledean_dist(u, v))

def add_vector(u, v):
    return [a+b for a, b in zip(u, v)]

def get_vector_uv(u, v):
    return tuple([b-a for a, b in zip(u, v)])

def get_nearest(p, locs):
    nearest = None
    nearest_d = sys.maxsize

    for l in locs:
        d = get_distance(p, l)
        if nearest_d > d:
            nearest_d = d
            nearest = l

    return nearest

def get_nearests(p, locs, margin=0, ensure_not_empty=True):
    nearest = None
    nearest_d = sys.maxsize
    nearests = []
    for l in locs:
        d = get_distance(p, l)
        if d < nearest_d:
            nearest_d = d
            nearest = l
        if d <= margin:
            nearests.append(l)
    if ensure_not_empty and len(nearests) == 0:
        nearests.append(nearest)
    return nearests

def get_nearests_to_radius(point, locs, radius, margin=1000):
    nearest = None
    nearest_d = sys.maxsize
    nearests = []
    for l in locs:
        d = get_distance(point, l)
        d = abs(d-radius)
        if d < nearest_d:
            nearest_d = d
            nearest = l
        if d <= margin:
            nearests.append(l)
    if len(nearests) == 0:
        nearests.append(nearest)
    assert(len(nearests))
    return nearests

def get_nearest_with_vectors(soma_loc, vectors, locs, margin,
        ensure_not_empty=True):
    assert len(vectors)
    v = random.choice(vectors)
    rosette_loc = [a+b for a, b in zip(soma_loc, v)]
    return get_nearests(rosette_loc, locs, margin,
                        ensure_not_empty=ensure_not_empty)

class GrC():

    def __init__(self, soma_loc):
        self.soma_loc = soma_loc
        self.num_claws_gt = None
        self.gt_claw_lens = []
        self.gt_claw_vectors = []
        self.clear_edges()

    def add_edge(self, mf_id, mf_loc):
        mf_loc = tuple(mf_loc)
        self.mfs.add(mf_id)
        self.mf_locs.append(tuple(mf_loc))
        self.edges.append((mf_id, tuple(mf_loc)))
        self.num_claws += 1

    def clear_edges(self):
        self.num_claws = 0
        self.mfs = set()
        self.mf_locs = []
        self.edges = []
        self.claw_vectors = []

    def compute_claw_vectors(self):
        claw_vectors = []
        for edge in self.edges:
            mf_loc = edge[1]
            claw_vectors.append(get_vector_uv(self.soma_loc, mf_loc))
        return claw_vectors

    def remove_dendrites(self, min_len=None, max_len=None):
        removes = []
        if min_len:
            for mf_loc in self.mf_locs:
                if get_eucledean_dist(self.soma_loc, mf_loc) < min_len:
                    removes.append(mf_loc)
        if max_len:
            for mf_loc in self.mf_locs:
                if get_eucledean_dist(self.soma_loc, mf_loc) > max_len:
                    removes.append(mf_loc)

        prev_edges = self.edges

        self.num_claws_gt = 0
        self.gt_claw_lens = []
        self.gt_claw_vectors = []
        self.num_claws = 0
        self.mfs = set()
        self.mf_locs = []
        self.edges = []
        self.claw_vectors = [] 

        for mf_id, mf_loc in prev_edges:
            if mf_loc in removes:
                continue
            self.num_claws_gt += 1
            self.num_claws += 1
            claw_vector = [a-b for a, b, in zip(mf_loc, self.soma_loc)]
            self.gt_claw_lens.append(get_eucledean_dist(self.soma_loc, mf_loc))
            self.claw_vectors.append(claw_vector)
            self.gt_claw_vectors.append(claw_vector)
            self.mfs.add(mf_id)
            self.mf_locs.append(mf_loc)
            self.edges.append((mf_id, mf_loc))


    def compute_rosette_distances(self, locs):
        # return
        distances = []
        for l in locs:
            d = get_distance(self.soma_loc, l)
            distances.append((d, l))
        self.sorted_distances = sorted(distances)

    def get_nearest_mf_locs(self, d, margin, filter_locs=None):
        res = []
        nearest = None
        nearest_d = sys.maxsize
        for bouton_dist, rosette_loc in self.sorted_distances:
            if filter_locs is not None and rosette_loc not in filter_locs:
                continue
            diff = abs(bouton_dist - d)
            if diff < nearest_d:
                nearest_d = diff
                nearest = rosette_loc
            if diff <= margin:
                res.append(rosette_loc)

        if len(res):
            return res
        else:
            # nothing within margin, return the nearest to d
            return [nearest]

    def get_capacity(self):
        c = self.num_claws_gt - self.num_claws
        if c < 0: c = 0
        return c

    def get_pattern(self):
        pattern = []
        for mf_id in sorted(self.mfs):
            pattern.append(mf_id)
        return pattern


class MF():

    def __init__(self, locs):
        self.locs = [tuple(l) for l in locs]
        self.clear_edges()
        self.num_claws = 0
        self.num_claws_gt = 0
        self.edge_lengths_gt = []
        self.edge_lengths = []

    def add_edge(self, mf_loc, grc_id, grc_loc):
        self.grcs.add(grc_id)
        # self.claws.append((grc_id, tuple(mf_loc)))
        # self.claws.append((tuple(mf_loc), grc_id))
        self.claws[mf_loc].append(grc_id)
        self.edge_lengths.append(
            get_distance(mf_loc, grc_loc))
        self.num_claws += 1

    def clear_edges(self):
        self.grcs = set()
        self.edge_lengths = []
        self.num_claws = 0
        self.claws = {}
        for loc in self.locs:
            self.claws[loc] = []

    def finalize_gt(self):
        self.num_claws_gt = self.num_claws
        self.claws_gt = self.claws
        self.edge_lengths_gt = self.edge_lengths

    def get_capacity(self):
        cap = self.num_claws_gt - self.num_claws
        if cap < 0: cap = 0
        return cap

    def get_mf_loc_capacity(self, mf_loc, size=False, gt=False, current=False):
        assert size or gt or current
        cap = len(self.claws_gt[mf_loc])
        if gt:
            return cap
        else:
            if size:
                return len(self.claws[mf_loc])
            cap -= len(self.claws[mf_loc])
            if cap < 0: cap = 0
            return cap

    def get_all_mf_locs_size(self, ret=None, gt=False):
        # cap = len(self.claws_gt[mf_loc])
        if gt:
            claws = self.claws_gt
        else:
            claws = self.claws
        if ret is None:
            ret = {}
        for loc in self.claws_gt:
            ret[loc] = len(self.claws_gt[loc])
        return ret

    # def get_rosette_loc_capacity(self, ret=None, gt=False):
    #     if gt:
    #         claws = self.claws_gt
    #     else:
    #         claws = self.claws
    #     if ret is None:
    #         ret = dict()
    #     for loc in self.locs:
    #         ret[loc] = 0
    #     for claw in claws:
    #         ret[claw[0]] += 1
    #     return ret

    def compute_grc_distances(self, locs):
        distances = []
        for l in locs:
            d = get_distance(self.soma_loc, l)
            distances.append((d, l))
        self.sorted_distances = sorted(distances)


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


class GCLGraph():

    def __init__(self, ):
        self.grcs = {}
        self.mfs = {}
        self.mf_locs = []
        self.grc_locs = []
        self.gt_claw_len_array = []
        self.gt_claw_vectors = []
        self.mf_loc_to_mf_id = {}
        self.grc_locs_to_id = {}

        self.synapse_to_bouton_max_dist = 22000  # 10um, an MF bouton can be pretty big
        self.min_synapses_per_bouton = 3
        self.avg_edge_count_by_depth = None
        self.avg_dendrite_len = None
        self.inferred_edges_joint_probability = None
        self.grc_xyzs = set()

    def add_mfs(self, mfs_locs, xlim=None, ylim=None, zlim=None):

        for mf_id in mfs_locs:
            locs = mfs_locs[mf_id]
            locs = [tuple(l) for l in locs]

            filter_list = []
            for loc in locs:
                if xlim and (loc[0] < xlim[0] or loc[0] > xlim[1]):
                    print(f'Skip OOB {loc}')
                    continue
                if ylim and (loc[1] < ylim[0] or loc[0] > ylim[1]):
                    print(f'Skip OOB {loc}')
                    continue
                if zlim and (loc[2] < zlim[0] or loc[0] > zlim[1]):
                    print(f'Skip OOB {loc}')
                    continue
                filter_list.append(loc)
            locs = filter_list

            if len(locs) == 0:
                print(f"Skipping {mf_id} because it has no boutons")
                continue
            
            mf = MF(locs)
            self.mfs[mf_id] = mf

            self.mf_locs.extend(locs)
            for loc in locs:
                self.mf_loc_to_mf_id[tuple(loc)] = mf_id

    def add_grc(self, grc_id, grc_xyz, mf_synapse_locs,
            compute_rosette_distances=True,
            min_synapses_per_bouton=None,
            verbose=None,
            mf_whitelist=None,
            synapse_whitelist=None,
            min_claws=1,
            ):

        if min_synapses_per_bouton is None:
            min_synapses_per_bouton = self.min_synapses_per_bouton
        if mf_whitelist is None:
            mf_whitelist = set()

        if verbose is None:
            verbose = 2

        grc_xyz = tuple(grc_xyz)
        if grc_xyz in self.grc_xyzs:
            print(f'Trying to add {grc_id} but {self.grc_locs_to_id[grc_xyz]} already exists at {grc_xyz}')
            assert False

        self.grc_xyzs.add(grc_xyz)
        grc = GrC(grc_xyz)
        self.grc_locs_to_id[grc_xyz] = grc_id
        self.grc_locs.append(grc_xyz)
        bouton_syn_count = collections.defaultdict(int)
        bouton_syns = collections.defaultdict(list)
        loc_to_mf = {}

        for mf_id, syn_loc in mf_synapse_locs:
            if mf_id not in self.mfs:
                print(
                    f"Skipping {grc_id} synapse at {to_ng_coord(syn_loc)} because"
                    f" {mf_id} is not in db"
                    )
                continue
            rosette_loc = tuple(get_nearest(syn_loc, self.mfs[mf_id].locs))
            syn_bouton_dist = get_distance(syn_loc, rosette_loc)
            if syn_bouton_dist > self.synapse_to_bouton_max_dist:
                print(f"Skipping {grc_id} {mf_id} synapse at {to_ng_coord(syn_loc)} to bouton loc {to_ng_coord(rosette_loc)} because dist {syn_bouton_dist} exceeds {self.synapse_to_bouton_max_dist}")
                continue

            loc_to_mf[rosette_loc] = mf_id
            bouton_syn_count[rosette_loc] += 1
            bouton_syns[rosette_loc].append(syn_loc)

        num_claws = 0
        min_synapses_per_bouton_skipped = 0
        for rosette_loc in bouton_syn_count:
            if bouton_syn_count[rosette_loc] < min_synapses_per_bouton:
                mf_id = loc_to_mf[rosette_loc]
                if mf_id in mf_whitelist:
                    print(f"mf_whitelist: {mf_id}")
                if mf_id not in mf_whitelist:
                    # check if synapse is white listed
                    whitelisted = False
                    for syn_loc in bouton_syns[rosette_loc]:
                        syn_loc = to_ng_coord(syn_loc)
                        if syn_loc in synapse_whitelist:
                            whitelisted = True
                            print(f'syn {syn_loc} whitelisted')
                    if not whitelisted:
                        if bouton_syn_count[rosette_loc] >= verbose:
                            print(f"Skipping {grc_id} {mf_id} synapses to bouton loc {to_ng_coord(rosette_loc)} because less than {min_synapses_per_bouton} synapses")
                            print([to_ng_coord(loc) for loc in bouton_syns[rosette_loc]])
                            # print(bouton_syns[rosette_loc])
                        min_synapses_per_bouton_skipped += 1
                        continue

            self.add_edge(rosette_loc, grc_id, grc=grc)
            gt_claw_len = get_distance(rosette_loc, grc.soma_loc)
            gt_claw_len = int(gt_claw_len)
            self.gt_claw_len_array.append(gt_claw_len)
            grc.gt_claw_lens.append(gt_claw_len)
            claw_vector = [a-b for a, b, in zip(rosette_loc, grc.soma_loc)]
            grc.gt_claw_vectors.append(claw_vector)
            self.gt_claw_vectors.append(claw_vector)
            num_claws += 1

        if min_synapses_per_bouton_skipped and num_claws <= 1:
            for mf_id, syn_loc in mf_synapse_locs:
                print(f'{mf_id}: {to_ng_coord(syn_loc)}', end=', ')
            print()
            if grc_id in ['grc_309', 'grc_335', 'grc_981', 'grc_757', 'grc_2765', 'grc_2123', 'grc_2131', 'grc_748', 'grc_1544', 'grc_478', 'grc_2287', 'grc_2364', 'grc_286', 'grc_533', 'grc_2158', 'grc_2475', 'grc_335', 'grc_335', 'grc_335', 'grc_335', 'grc_335', 'grc_335', 'grc_335', 'grc_335', 'grc_335', 'grc_335']:
                return
            print(f'{grc_id} has {num_claws} valid claws, skipping')
            print()
            print(f'processed {len(self.grcs)} grcs')
            asdf

        if num_claws < min_claws:
            for mf_id, syn_loc in mf_synapse_locs:
                print(f'{mf_id}: {to_ng_coord(syn_loc)}', end=', ')
            print()
            print(f'{grc_id} has {num_claws} valid claws, skipping')
            return

        grc.num_claws_gt = num_claws
        if compute_rosette_distances:
            grc.compute_rosette_distances(self.mf_locs)

        self.grcs[grc_id] = grc

    def clear_graph(self):
        for grc in self.grcs:
            self.grcs[grc].clear_edges()
        for mf in self.mfs:
            self.mfs[mf].clear_edges()

    def finalize_gt(self):
        for mf in self.mfs:
            self.mfs[mf].finalize_gt()
        self.mf_locs_set = set(self.mf_locs)

    def randomize_graph(
            self,
            grc_local=False,
            preserve_mf_out_degree=False,
            mf_dist_margin=1000,
            winner=False,
            use_claw_vectors=False,
            remove_empty=False,
            random_model=False,
            constant_grc_degree=False,
            dendrite_range=None,
            ):

        self.clear_graph()

        if random_model:
            if constant_grc_degree:
                num_claws = constant_grc_degree
            else:
                num_claws = 4
            if random_model == "all_mfs":
                mf_ids = [k for k in self.mfs.keys()]
            else:
                mf_ids = [k for k in self.mfs.keys() if self.mfs[k].num_claws_gt]
            for grc_id in self.grcs:
                grc = self.grcs[grc_id]
                while grc.num_claws < num_claws:
                    i = int(random.random()*len(mf_ids))
                    mf_loc = self.mfs[mf_ids[i]].locs[0]
                    self.add_edge(mf_loc, grc_id)
            return

        for grc_id in self.grcs:
            grc = self.grcs[grc_id]
            while (grc.num_claws < grc.num_claws_gt):

                if dendrite_range:
                    assert False, 'Untested'
                    claw_len = (dendrite_range[0] + dendrite_range[1]) / 2
                    margin = abs(dendrite_range[1]-claw_len)
                    potential_rosettes = grc.get_nearest_mf_locs(
                        claw_len, margin=margin)
                elif not use_claw_vectors:
                    if grc_local:
                        claw_len = self.get_random_claw_length(grc.gt_claw_lens)
                    else:
                        claw_len = self.get_random_claw_length()
                    potential_rosettes = grc.get_nearest_mf_locs(
                        claw_len, margin=mf_dist_margin)
                else:
                    assert not grc_local
                    claw_vectors = self.gt_claw_vectors
                    potential_rosettes = get_nearest_with_vectors(
                        soma_loc=grc.soma_loc,
                        vectors=claw_vectors,
                        locs=self.mf_locs,
                        margin=mf_dist_margin,
                        )

                if winner:
                    new_list = []
                    for loc in potential_rosettes:
                        mf_id = self.mf_loc_to_mf_id[tuple(loc)]
                        mf = self.mfs[mf_id]
                        assert False
                        connected = len(mf.claws) + 1
                        new_list.extend([tuple(loc)]*connected)
                    if len(new_list):
                        potential_rosettes = new_list

                if preserve_mf_out_degree:
                    new_list = []
                    for loc in potential_rosettes:
                        mf_id = self.mf_loc_to_mf_id[tuple(loc)]
                        mf = self.mfs[mf_id]
                        capacity = mf.get_capacity()
                        if capacity:
                            new_list.extend([tuple(loc)]*capacity)
                    if len(new_list):
                        potential_rosettes = new_list

                idx = int(random.random()*len(potential_rosettes))
                rosette_loc = potential_rosettes[idx]
                self.add_edge(rosette_loc, grc_id)

    def randomize_graph_by_mf(
            self,
            # grc_local=False,
            # preserve_mf_out_degree=False,
            mf_dist_margin=1000,
            local_lengths=False,
            single_connection_per_pair=False,
            preserve_in_degree=False,
            approximate_in_degree=False,
            # winner=False,
            # use_claw_vectors=False,
            # remove_empty=False,
            ):
        assert not (preserve_in_degree and approximate_in_degree)
        self.clear_graph()

        mf_locs = []
        for mf_id in self.mfs:
            mf = self.mfs[mf_id]
            claw_per_loc = mf.get_all_mf_locs_size(gt=True)
            for loc, count in claw_per_loc.items():
                mf_locs.extend([loc]*count)

        connected = set()
        while len(mf_locs):
            locs_idx = int(random.random()*len(mf_locs))
            mf_loc = mf_locs.pop(locs_idx)
            if local_lengths:
                mf = self.mf_loc_to_mf_id[mf_loc]
                mf = self.mfs[mf]
                claw_len = self.get_random_claw_length(mf.edge_lengths_gt)
            else:
                claw_len = self.get_random_claw_length()
            avail_grcs_locs = get_nearests_to_radius(
                    point=mf_loc,
                    locs=self.grc_locs,
                    radius=claw_len,
                    margin=mf_dist_margin)

            if single_connection_per_pair:
                single_avail_grcs_locs = [g for g in avail_grcs_locs if (mf_loc, g) not in connected]
                if len(single_avail_grcs_locs):
                    avail_grcs_locs = single_avail_grcs_locs

            if preserve_in_degree:
                new_list = []
                for loc in avail_grcs_locs:
                    grc = self.grcs[self.grc_locs_to_id[loc]]
                    capacity = grc.get_capacity()
                    if capacity:
                        new_list.extend([tuple(loc)]*capacity)
                if len(new_list):
                    avail_grcs_locs = new_list

            if approximate_in_degree:
                new_list = []
                for loc in avail_grcs_locs:
                    grc = self.grcs[self.grc_locs_to_id[loc]]
                    capacity = grc.num_claws_gt
                    if capacity:
                        new_list.extend([tuple(loc)]*capacity)
                if len(new_list):
                    avail_grcs_locs = new_list

            grc_idx = int(random.random()*len(avail_grcs_locs))
            grc_loc = avail_grcs_locs[grc_idx]
            connected.add((mf_loc, grc_loc))
            self.add_edge(mf_loc, self.grc_locs_to_id[grc_loc])

    def randomize_graph_by_grc(
            self,
            mf_dist_margin=1000,
            local_lengths=False,
            preserve_mf_degree=False,
            approximate_mf_degree=False,
            single_connection_per_pair=False,
            constant_grc_degree=None,
            constant_dendrite_length=False,
            always_pick_closest_rosette=False,
            dendrite_range=None,
            ):

        self.clear_graph()

        if constant_grc_degree:
            num_claw_per_grc = len(self.gt_claw_len_array) / len(self.grcs)
            print(f"Average num_claw_per_grc: {num_claw_per_grc}")
            # if not isinstance(constant_grc_degree, int):
            #     num_claw_per_grc = int(num_claw_per_grc + .5)
            #     print(f"Rounding num_claw_per_grc to {num_claw_per_grc}")
            # else:
            #     num_claw_per_grc = constant_grc_degree
            if not isinstance(constant_grc_degree, bool):
                num_claw_per_grc = constant_grc_degree
        grcs = []
        for grc_id in self.grcs:
            grc = self.grcs[grc_id]
            if constant_grc_degree:
                # print(num_claw_per_grc)
                if isinstance(num_claw_per_grc, int):
                    grcs.extend([grc_id]*num_claw_per_grc)
                elif isinstance(num_claw_per_grc, float):
                    num_claw_per_grc_int = int(num_claw_per_grc)
                    if (num_claw_per_grc_int + random.random()) < num_claw_per_grc:
                        num_claw_per_grc_int += 1
                    grcs.extend([grc_id]*num_claw_per_grc_int)
                else:
                    assert False
            else:
                grcs.extend([grc_id]*grc.num_claws_gt)

        connected = set()
        while len(grcs):
            grcs_idx = int(random.random()*len(grcs))
            grc_id = grcs.pop(grcs_idx)
            grc = self.grcs[grc_id]
            if constant_grc_degree is None:
                assert grc.num_claws < grc.num_claws_gt

            if dendrite_range:
                claw_len = (dendrite_range[0] + dendrite_range[1]) / 2
                margin = abs(dendrite_range[1]-claw_len)
                potential_mf_locs = grc.get_nearest_mf_locs(
                    claw_len, margin=margin)
            else:
                if constant_dendrite_length:
                    claw_len = constant_dendrite_length
                elif local_lengths:
                    claw_len = self.get_random_claw_length(grc.gt_claw_lens)
                else:
                    claw_len = self.get_random_claw_length()
                potential_mf_locs = grc.get_nearest_mf_locs(
                    claw_len, margin=mf_dist_margin)

            if single_connection_per_pair:
                new_list = [mf_loc for mf_loc in potential_mf_locs if (mf_loc, grc_id) not in connected]
                if len(new_list):
                    potential_mf_locs = new_list

            if preserve_mf_degree:
                new_list = []
                for mf_loc in potential_mf_locs:
                    capacity = self.get_mf_loc_capacity(mf_loc)
                    if capacity:
                        new_list.extend([tuple(mf_loc)]*capacity)
                if len(new_list):
                    potential_mf_locs = new_list

            if approximate_mf_degree:
                new_list = []
                for mf_loc in potential_mf_locs:
                    capacity = self.get_mf_loc_capacity(mf_loc, gt=True)
                    if capacity:
                        new_list.extend([tuple(mf_loc)]*capacity)
                if len(new_list):
                    potential_mf_locs = new_list

            if always_pick_closest_rosette:
                potential_mf_locs = set(potential_mf_locs)
                mf_loc = get_nearest(grc.soma_loc, potential_mf_locs)
            else:
                # choose one by random (weighted)
                idx = int(random.random()*len(potential_mf_locs))
                mf_loc = potential_mf_locs[idx]
            connected.add((mf_loc, grc_id))
            self.add_edge(mf_loc, grc_id)

    def calculate_avg_dendrite_len(self):
        avg = int(sum(self.gt_claw_len_array)/len(self.gt_claw_len_array))
        self.avg_dendrite_len = avg

    def calculate_avg_edge_count_by_depth(self, bucket_size=2500):
        avg_edge_count_by_depth_sum = defaultdict(list)
        for grc_id in self.grcs:
            z =  self.grcs[grc_id].soma_loc[2]
            c = self.grcs[grc_id].num_claws_gt
            avg_edge_count_by_depth_sum[int(z/bucket_size)].append(c)
        # print(avg_edge_count_by_depth_sum); asdf
        self.avg_edge_count_by_depth = {}
        for k in avg_edge_count_by_depth_sum:
            avg = sum(avg_edge_count_by_depth_sum[k])/len(avg_edge_count_by_depth_sum[k])
            self.avg_edge_count_by_depth[k] = avg
        # for k in sorted(self.avg_edge_count_by_depth.keys()):
        #     print(f'{k}: {self.avg_edge_count_by_depth[k]} ({len(avg_edge_count_by_depth_sum[k])})')
        # print(self.avg_edge_count_by_depth); asdf

    def get_avg_edge_count_by_depth(self, loc, bucket_size=2500):
        k = int(loc[2]/bucket_size)
        return self.avg_edge_count_by_depth[k]

    def expand_locs_by_size(self, locs):
        # print(locs)
        # print(new_list)
        # print(self.mf_loc_to_mf_id)
        new_list = []
        for mf_loc in locs:
            mf_loc = tuple(mf_loc)
            capacity = self.get_mf_loc_capacity(mf_loc, gt=True)
            if capacity:
                new_list.extend([mf_loc]*capacity)
        if len(new_list):
            locs = new_list
        # print(new_list)
        return locs


    def randomize_graph_by_grc2(
            self,
            mf_dist_margin=5000,
            local_lengths=False,
            preserve_mf_degree=False,
            approximate_mf_degree=False,
            single_connection_per_pair=True,
            constant_grc_degree=None,
            constant_dendrite_length=False,
            always_pick_closest_rosette=False,
            dendrite_range=None,
            seed=None,
            ):

        self.clear_graph()

        if seed is not None:
            random.seed(seed)

        if constant_grc_degree:
            num_claw_per_grc = len(self.gt_claw_len_array) / len(self.grcs)
            print(f"Average num_claw_per_grc: {num_claw_per_grc}")
            if not isinstance(constant_grc_degree, bool):
                num_claw_per_grc = constant_grc_degree

        if constant_dendrite_length:
            if isinstance(constant_dendrite_length, bool):
                if self.avg_dendrite_len is None:
                    self.calculate_avg_dendrite_len()
                constant_dendrite_length = self.avg_dendrite_len
                print(f"Average dendrite len: {self.avg_dendrite_len}")
            assert isinstance(constant_dendrite_length, int)

        grcs = []
        for grc_id in self.grcs:
            grc = self.grcs[grc_id]
            if constant_grc_degree == 'depth':
                if self.avg_edge_count_by_depth is None:
                    self.calculate_avg_edge_count_by_depth()
                c = self.get_avg_edge_count_by_depth(grc.soma_loc)
                c = int(c+random.random())
                grcs.extend([grc_id]*c)
            elif constant_grc_degree:
                if isinstance(num_claw_per_grc, int):
                    grcs.extend([grc_id]*num_claw_per_grc)
                elif isinstance(num_claw_per_grc, float):
                    num_claw_per_grc_int = int(num_claw_per_grc+random.random())
                    grcs.extend([grc_id]*num_claw_per_grc_int)
                else:
                    assert False
            else:
                grcs.extend([grc_id]*grc.num_claws_gt)

        random.shuffle(grcs)
        connected = set()
        while len(grcs):
            grc_id = grcs.pop()
            grc = self.grcs[grc_id]
            if constant_grc_degree is None:
                assert grc.num_claws < grc.num_claws_gt

            if dendrite_range:
                claw_len = (dendrite_range[0] + dendrite_range[1]) / 2
                margin = abs(dendrite_range[1]-claw_len)
                potential_mf_locs = set(grc.get_nearest_mf_locs(
                    claw_len, margin=margin, filter_locs=self.mf_locs_set))
            else:
                if constant_dendrite_length:
                    claw_len = constant_dendrite_length
                elif local_lengths:
                    claw_len = self.get_random_claw_length(grc.gt_claw_lens)
                else:
                    claw_len = self.get_random_claw_length()
                potential_mf_locs = set(grc.get_nearest_mf_locs(
                    claw_len, margin=mf_dist_margin, filter_locs=self.mf_locs_set))

            if single_connection_per_pair:
                new_list = [mf_loc for mf_loc in potential_mf_locs if (mf_loc, grc_id) not in connected]
                if len(new_list):
                    potential_mf_locs = new_list

            if preserve_mf_degree:
                new_list = []
                for mf_loc in potential_mf_locs:
                    capacity = self.get_mf_loc_capacity(mf_loc)
                    if capacity:
                        new_list.extend([tuple(mf_loc)]*capacity)
                if len(new_list):
                    potential_mf_locs = new_list

            if approximate_mf_degree:
                new_list = []
                for mf_loc in potential_mf_locs:
                    capacity = self.get_mf_loc_capacity(mf_loc, gt=True)
                    if capacity:
                        new_list.extend([tuple(mf_loc)]*capacity)
                if len(new_list):
                    potential_mf_locs = new_list

            if always_pick_closest_rosette:
                potential_mf_locs = potential_mf_locs
                mf_loc = get_nearest(grc.soma_loc, potential_mf_locs)
            else:
                # choose one by random (weighted)
                potential_mf_locs = list(potential_mf_locs)
                idx = int(random.random()*len(potential_mf_locs))
                mf_loc = potential_mf_locs[idx]
            connected.add((mf_loc, grc_id))
            self.add_edge(mf_loc, grc_id)

    def shuffle_edges(
            self,
            mf_dist_margin=5000,
            max_edge_len=None,
            remove_empty=False,
            approximate_mf_degree=False,
            use_global_distribution=False,
            seed=None,
            ):

        if not use_global_distribution and self.edge_probabilities_by_pos is None:
            self.build_edge_probability_given_position(max_edge_len=max_edge_len)

        self.clear_graph()
        if seed is not None:
            random.seed(seed)

        connected = set()
        # grc_vectors = {}
        grcs = []
        for grc_id in self.grcs:
            grc = self.grcs[grc_id]
            grcs.extend([grc_id]*grc.num_claws_gt)
            # grc_vectors[grc_id] = copy.deepcopy(grc.gt_claw_vectors)

        while len(grcs):
            grcs_idx = int(random.random()*len(grcs))
            grc_id = grcs.pop(grcs_idx)
            grc = self.grcs[grc_id]
            assert grc.num_claws < grc.num_claws_gt

            if not use_global_distribution:
                claw_vectors = self.get_possible_edges_given_position(grc)
            else:
                claw_vectors = self.gt_claw_vectors

            potential_mf_locs = get_nearest_with_vectors(
                soma_loc=grc.soma_loc,
                vectors=claw_vectors,
                locs=self.mf_locs,
                margin=mf_dist_margin,
                ensure_not_empty=False,
                )

            # make sure single connection per pair
            potential_mf_locs = [mf_loc for mf_loc in potential_mf_locs if (mf_loc, grc_id) not in connected]

            if len(potential_mf_locs) == 0:
                grcs.append(grc_id)
                continue

            if remove_empty:
                assert False, "Untested?"
                potential_mf_locs_new = []
                for rosette in potential_mf_locs:
                    mf_id = self.mf_loc_to_mf_id[tuple(rosette)]
                    mf = self.mfs[mf_id]
                    if mf.num_claws_gt:
                        potential_mf_locs_new.append(rosette)
                if len(potential_mf_locs_new):
                    potential_mf_locs = potential_mf_locs_new

            if approximate_mf_degree:
                new_list = []
                for mf_loc in potential_mf_locs:
                    capacity = self.get_mf_loc_capacity(mf_loc, gt=True)
                    if capacity:
                        new_list.extend([tuple(mf_loc)]*capacity)
                if len(new_list):
                    potential_mf_locs = new_list

            rosette_loc = random.choice(potential_mf_locs)
            self.add_edge(rosette_loc, grc_id)
            connected.add((rosette_loc, grc_id))

    def randomize(self,
            preserve_mf_degree=False,
            seed=None,
            ):
        self.clear_graph()
        if seed is not None:
            random.seed(seed)

        assert preserve_mf_degree is False or preserve_mf_degree == 'strict' or preserve_mf_degree == 'soft'

        grcs = []
        for grc_id in self.grcs:
            grcs.extend([grc_id]*self.grcs[grc_id].num_claws_gt)
        random.shuffle(grcs)

        if preserve_mf_degree is not False:
            mf_locs = []
            for loc in self.mf_locs:
                capacity = self.get_mf_loc_capacity(loc, gt=True)
                mf_locs.extend([loc]*capacity)
            # random.shuffle(mf_locs)
        else:
            mf_locs = copy.deepcopy(self.mf_locs)
        random.shuffle(mf_locs)

        while len(grcs):
            grc_id = grcs.pop()
            grc = self.grcs[grc_id]
            i = int(random.random()*len(mf_locs))
            loc = mf_locs[i]
            self.add_edge(loc, grc_id)
            if preserve_mf_degree == 'strict':
                mf_locs.pop(i)


    def get_random_claw_length(self, array=None):
        if array is None:
            array = self.gt_claw_len_array
        idx = int(random.random()*len(array))
        return array[idx]

    def add_edge(self, mf_loc, grc_id, grc=None):

        if grc is None:
            grc = self.grcs[grc_id]
        mf_loc = tuple(mf_loc)
        grc_loc = tuple(grc.soma_loc)

        mf_id = self.mf_loc_to_mf_id[mf_loc]
        mf = self.mfs[mf_id]
        mf.add_edge(mf_loc, grc_id=grc_id, grc_loc=grc_loc)
        grc.add_edge(mf_id, mf_loc)

    def count_unique_patterns(self):
        patterns = set()
        for grc_id in self.grcs:
            patterns.add(tuple(self.grcs[grc_id].get_pattern()))
        return len(patterns)

    def count_unsampled_mfs(self):
        count = 0
        for mf_id, mf in self.mfs.items():
            if mf.num_claws == 0:
                count += 1
        return count

    def print_mfs_connected_summary(self):
        count = defaultdict(int)
        for mf_id, mf in self.mfs.items(): 
            count[mf.num_claws] += 1
        for i in [1, 2, 3, 4]:
            print(f'{i}: {count[i]}')

    def get_mf_loc_capacity(self, mf_loc, size=False, gt=False):
        mf_id = self.mf_loc_to_mf_id[mf_loc]
        return self.mfs[mf_id].get_mf_loc_capacity(mf_loc, size, gt)

    def remove_empty_mfs(self):
        ret = []
        for mf_id, mf in self.mfs.items(): 
            remove_boutons = []
            for loc in mf.locs:
                if len(mf.claws[loc]) == 0:
                    remove_boutons.append(loc)
            for loc in remove_boutons:
                mf.locs.remove(loc)
                mf.claws_gt.pop(loc)
                self.mf_locs.remove(loc)
                self.mf_locs_set.remove(loc)
                # mf.claws.pop(loc)
            if len(remove_boutons):
                remove_boutons = [to_ng_coord(coord) for coord in remove_boutons]
                ret.append((mf_id, remove_boutons))

        remove_mfs = []
        for mf_id, mf in self.mfs.items(): 
            if mf.num_claws == 0:
                remove_mfs.append(mf_id)
        for mf_id in remove_mfs:
            self.mfs.pop(mf_id)

        ret.extend(remove_mfs)
        return ret

    def build_inferred_edges_joint_probability(self,
            bottom_margins,
            top_margins,
            height_bucket_size,
            ):
        bottom_probabilities = defaultdict(list)
        top_probabilities = defaultdict(list)

        for grc_id, grc in self.grcs.items():
            height_id = int(grc.soma_loc[1]/height_bucket_size)
            z_pos = grc.soma_loc[2]
            if (z_pos > bottom_margins[0] and z_pos < bottom_margins[1]):
                bottom_probabilities[height_id].append(grc.compute_claw_vectors())
            if (z_pos > top_margins[0] and z_pos < top_margins[1]):
                top_probabilities[height_id].append(grc.compute_claw_vectors())

        self.inferred_edges_joint_probability = (bottom_probabilities, top_probabilities, height_bucket_size)
        for k in sorted(self.inferred_edges_joint_probability[0].keys()):
            v = self.inferred_edges_joint_probability[0][k]
            print(f'{k}: {len(v)}')
        for k in sorted(self.inferred_edges_joint_probability[1].keys()):
            v = self.inferred_edges_joint_probability[1][k]
            print(f'{k}: {len(v)}')

    def get_possible_edges(self, probs, height_id):
        if height_id not in probs:
            if height_id+1 in probs:
                height_id += 1
            else:
                height_id -= 1
        return random.choice(probs[height_id])

    def build_edge_probability_given_position(self, max_edge_len=None):
        bucket_size = 10000
        self.edge_probabilities_by_pos = defaultdict(lambda: defaultdict(list))
        self.edge_probabilities_by_pos_all = []
        for grc_id, grc in self.grcs.items():
            height_id = int(grc.soma_loc[1]/bucket_size)
            depth_id = int(grc.soma_loc[2]/bucket_size)
            for e in grc.compute_claw_vectors():
                if max_edge_len and np.linalg.norm(e) > max_edge_len:
                    # print(f'skipping {e}')
                    continue
                self.edge_probabilities_by_pos[height_id][depth_id].append(e)
                self.edge_probabilities_by_pos_all.append(e)

    def get_possible_edges_given_position(self, grc):
        bucket_size = 10000
        height_id = int(grc.soma_loc[1]/bucket_size)
        depth_id = int(grc.soma_loc[2]/bucket_size)
        # return random.choice(self.edge_probabilities_by_pos[height_id][depth_id])
        ret = self.edge_probabilities_by_pos[height_id][depth_id]
        if len(ret) == 0:
            return self.edge_probabilities_by_pos_all
        else:
            return ret

    def replicate(
            self,
            bottom_margin,
            top_margin,
            n_replicates,
            wraps_z=True,
            seed=1234,
            ):
        random.seed(seed)
        assert self.inferred_edges_joint_probability is not None
            # self.build_inferred_edges_joint_probability()
        mf_dist_margin = 15000
        n_replicates -= 1
        assert n_replicates > 0

        replicate_width = top_margin - bottom_margin
        assert replicate_width > 0

        for grc_id, grc in self.grcs.items():
            assert grc.soma_loc[2] >= bottom_margin and grc.soma_loc[2] < top_margin, f'{grc_id} out of margin {bottom_margin}/{top_margin} at {grc.soma_loc}'

        # replicate mfs
        extended_mfss = {}
        new_mf_locs = []
        for i_rep in range(n_replicates):
            extended_mfs = []
            z_offset = replicate_width*(i_rep+1)
            for mf_id, mf in self.mfs.items():
                locs = mf.locs
                new_locs = []
                for loc in locs:
                    new_loc = [k for k in loc]
                    new_loc[2] += z_offset
                    new_locs.append(tuple(new_loc))
                new_mf = MF(new_locs)
                new_id = mf_id + f'_{i_rep+1}'
                extended_mfss[new_id] = new_mf
                for loc in new_locs:
                    self.mf_loc_to_mf_id[tuple(loc)] = new_id
                new_mf_locs.extend(new_locs)

        for mf_id, mf in extended_mfss.items():
            self.mfs[mf_id] = mf
        # self.mf_locs.extend(new_mf_locs)

        # replicate grcs
        new_grcs = {}
        for i_rep in range(n_replicates):
            z_offset = replicate_width*(i_rep+1)
            for grc_id, grc in self.grcs.items():
                new_grc_xyz = list(grc.soma_loc)
                new_grc_xyz[2] += z_offset
                new_grc_xyz = tuple(new_grc_xyz)
                new_grc_id = grc_id + f'_{i_rep+1}'
                new_grc = GrC(new_grc_xyz)
                self.grc_locs_to_id[new_grc_xyz] = new_grc_id
                self.grc_locs.append(new_grc_xyz)
                for rosette_loc in grc.mf_locs:
                    rosette_loc = list(rosette_loc)
                    rosette_loc[2] += z_offset
                    rosette_loc = tuple(rosette_loc)
                    self.add_edge(rosette_loc, new_grc_id, grc=new_grc)
                new_grc.num_claws_gt = grc.num_claws
                new_grcs[new_grc_id] = new_grc

        self.grcs.update(new_grcs)

        # print(f'top_margin: {top_margin} ({top_margin/40})')
        # print(f'bottom_margin: {bottom_margin} ({bottom_margin/40})')
        z_plus_added = 0
        z_minus_added = 0
        n_processed = 0
        # connect inferred edges
        bottom_probabilities, top_probabilities, height_bucket_size = self.inferred_edges_joint_probability
        for grc_id, grc in self.grcs.items():
            height_id = int(grc.soma_loc[1]/height_bucket_size)
            relative_z = (grc.soma_loc[2] - bottom_margin) % replicate_width
            dist_to_top = replicate_width - relative_z
            dist_to_bottom = -relative_z
            # print()
            # print(f'top_margin: {top_margin} ({top_margin/40})')
            # print(f'bottom_margin: {bottom_margin} ({bottom_margin/40})')
            # print(f'grc_id: {grc_id}')
            # print(f'grc.soma_loc: {grc.soma_loc}')
            # print(f'abs_z: {grc.soma_loc[2]} ({grc.soma_loc[2]/40})')
            # print(f'relative_z: {relative_z} ({relative_z/40})')
            # print(f'dist_to_top/bottom: {dist_to_top} ({dist_to_top/40}), {dist_to_bottom} ({dist_to_bottom/40})')

            # ({dist_to_bottom/40})
            assert dist_to_top > 0
            assert dist_to_bottom <= 0
            missing_z_plus = 0
            possible_edges = self.get_possible_edges(bottom_probabilities, height_id)
            for e in possible_edges:
                if e[2] > dist_to_top:
                    if self.add_inferred_edge(grc_id, e, (bottom_margin, top_margin), mf_dist_margin, n_replicates, wraps_z):
                        missing_z_plus += 1
            missing_z_minus = 0
            possible_edges = self.get_possible_edges(top_probabilities, height_id)
            for e in possible_edges:
                if e[2] < dist_to_bottom:
                    if self.add_inferred_edge(grc_id, e, (bottom_margin, top_margin), mf_dist_margin, n_replicates, wraps_z):
                        missing_z_minus += 1
            z_plus_added += missing_z_plus
            z_minus_added += missing_z_minus
            grc.num_claws_gt = grc.num_claws
            n_processed += 1
            if n_processed % 1000 == 0:
                print(f'Processed {n_processed} grcs')
        print(f'z_plus_added: {z_plus_added}')
        print(f'z_minus_added: {z_minus_added}')

        self.mf_locs.extend(new_mf_locs)
        for mf_id, mf in self.mfs.items():
            mf.finalize_gt()
        print(f'New graph size: {len(self.grcs)} grcs, {len(self.mfs)} mfs')

    def add_inferred_edge(
            self, grc_id, edge, rep_margins, mf_dist_margin, n_replicates, wraps_z):
        grc = self.grcs[grc_id]
        replicate_width = rep_margins[1] - rep_margins[0]
        bottom_bound = rep_margins[0]
        top_bound = rep_margins[1] + replicate_width*n_replicates
        # print(f'adding edge {edge} to {grc.soma_loc}')
        abs_loc = add_vector(edge, grc.soma_loc)
        # print(f'abs_loc: {abs_loc}')
        if abs_loc[2] < bottom_bound:
            if not wraps_z:
                return False
            else:
                # print(f'abs_loc: {abs_loc}')
                abs_loc[2] = abs_loc[2] - bottom_bound + top_bound
                # print(f'adj_abs_loc: {abs_loc}')
                # asdf
        elif abs_loc[2] > top_bound:
            if not wraps_z:
                return False
            else:
                # print(f'abs_loc: {abs_loc}')
                abs_loc[2] = abs_loc[2] - top_bound + bottom_bound
                # print(f'adj_abs_loc: {abs_loc}')
                # asdf
        rel_loc = list(abs_loc)
        rel_loc[2] = ((rel_loc[2] - rep_margins[0]) % replicate_width) + rep_margins[0]
        # print(f'rel_loc: {rel_loc}')
        margin = mf_dist_margin
        potential_rosettes = []
        while len(potential_rosettes) == 0:
            potential_rosettes = get_nearests(
                p=rel_loc,
                locs=self.mf_locs,
                margin=margin,
                )
            margin *= 2
        assert len(potential_rosettes)
        # get random mf by capacity
        potential_rosettes = self.expand_locs_by_size(potential_rosettes)
        rel_mf_loc = list(random.choice(potential_rosettes))
        # print(f'rel_mf_loc: {rel_mf_loc}')
        abs_mf_loc = rel_mf_loc
        abs_mf_loc[2] = rel_mf_loc[2] + (abs_loc[2]-rel_loc[2])
        abs_mf_loc = tuple(abs_mf_loc)
        # print(f'abs_mf_loc: {abs_mf_loc}')
        # print(f'to {self.mf_loc_to_mf_id[abs_mf_loc]}')
        self.add_edge(abs_mf_loc, grc_id)
        return True

    def remove_dendrites(self, min_len=None, max_len=None):
        for grc in self.grcs:
            self.grcs[grc].remove_dendrites(min_len, max_len)
        self.gt_claw_len_array = []
        for grc in self.grcs:
            self.gt_claw_len_array.extend(self.grcs[grc].gt_claw_lens)

    def make_redundant(self, redundant_factor, n_dendrite_share=2, seed=None):

        assert redundant_factor >= 1
        if redundant_factor == 1:
            return self
        if seed is not None:
            random.seed(seed)

        n_grcs = len(self.grcs)
        n_grcs_to_keep = int(n_grcs / redundant_factor +.5)
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
        n_grcs_to_add = n_grcs - n_grcs_to_keep
        for i in range(n_grcs_to_add):
            current_grc_id = current_grc_ids.get_one()
            current = new_grcs[current_grc_id]
            redundant_grc = copy.deepcopy(current)

            new_id = n_grcs_to_keep + i
            assert new_id not in new_grcs
            new_grcs[new_id] = redundant_grc

            # remove n dendrites and add n random dendrites
            edges = redundant_grc.edges
            n_dendrite_share = min(n_dendrite_share, len(edges))
            n_remove = len(edges) - n_dendrite_share
            random.shuffle(edges)
            for j in range(n_remove):
                edges.pop()
            for j in range(n_remove):
                claw_len = self.avg_dendrite_len
                mf_dist_margin = 5000
                potential_mf_locs = set(redundant_grc.get_nearest_mf_locs(
                    claw_len, margin=mf_dist_margin, filter_locs=self.mf_locs_set))
                potential_mf_locs = list(potential_mf_locs)
                idx = int(random.random()*len(potential_mf_locs))
                mf_loc = potential_mf_locs[idx]
                # self.add_edge(mf_loc, grc_id)
                redundant_grc.edges.append((self.mf_loc_to_mf_id[mf_loc], mf_loc))
                mf = self.mfs[self.mf_loc_to_mf_id[mf_loc]]
                mf.add_edge(mf_loc, grc_id=None, grc_loc=(0, 0, 0))

        assert len(new_grcs) == len(self.grcs)
        self.grcs = new_grcs
        return self



def shuffle(input_graph, model):

    if model == 'local_random':
        # assuming constant grc degree and dendrite length as from the input graph
        input_graph.randomize_graph_by_grc2(
            single_connection_per_pair=True,
            constant_grc_degree='depth',
            constant_dendrite_length=True,
    #         always_pick_closest_rosette=True,
            )
    else:
        raise RuntimeError(f"{model} is invalid")
