
import sys
from collections import defaultdict
import compress_pickle
import importlib
import numpy as np
import copy

import daisy

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')
sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/segway.dahlia')

import segway.dahlia.db_server


def to_ng(coord):
    return [
        int(coord[0]/4),
        int(coord[1]/4),
        int(coord[2]/40),
        ]

seg_file = "/n/f810/htem/Segmentation/cb2_v4/output.zarr"
seg = daisy.open_ds(seg_file, 'volumes/super_1x2x2_segmentation_0.500_mipmap/s2')

def get_segment_id(seg, loc):
    loc = daisy.Coordinate((loc[2], loc[1], loc[0]))
    return int(seg[loc])


def to_voxel(coord, voxel_size):
    return (int(coord[0] / voxel_size[0]),
            int(coord[1] / voxel_size[1]),
            int(coord[2] / voxel_size[2]))

def get_eucledean_dist(a, b, scale=(1, 1, 1)):
    a = [k*s for k, s in zip(a, scale)]
    b = [k*s for k, s in zip(b, scale)]
    return np.linalg.norm(
        (a[0]-b[0], a[1]-b[1], a[2]-b[2]))

def make_neurondb_server():
    db = segway.dahlia.db_server.NeuronDBServer(
        db_name='neurondb_cb2_v4',
        # host='mongodb://10.117.28.250:27018/',  # balin
        host='mongodb://10.117.28.139:27017/',  # dwalin
        )
    db.connect()
    return db

neuron_db = make_neurondb_server()

def generate_interpolates(p0, p1, n):
    n += 1
    p0 = daisy.Coordinate(p0)
    p1 = daisy.Coordinate(p1)
    delta = daisy.Coordinate(tuple([
            (p1[0]-p0[0])/n,
            (p1[1]-p0[1])/n,
            (p1[2]-p0[2])/n,
        ]))
    points = []
    for i in range(n):
        points.append(p0+delta*i)
    return points

def generate_interpolates(p0, p1, n):
    n += 1
    p0 = daisy.Coordinate(p0)
    p1 = daisy.Coordinate(p1)
    delta = daisy.Coordinate(tuple([
            (p1[0]-p0[0])/n,
            (p1[1]-p0[1])/n,
            (p1[2]-p0[2])/n,
        ]))
    points = []
    for i in range(n):
        points.append(p0+delta*i)
    return points

def within_dist(p0, p1, threshold, min_threshold, scale):
    d = get_eucledean_dist(p0, p1, scale=scale)
    if d > threshold:
        return False
    if min_threshold and d < min_threshold:
        return False
    return True


def get_near_locs(locs, seg,
        threshold=1100, scale=(1, 1, 1),
        min_threshold=None,
        ):
    nears = []
    # pre_locs, post_locs = sorted(list(set(post_locs)))
    locs = sorted(list(set(locs)))
    pre_locs, post_locs = zip(*locs)
    # print(post_locs)
    for i, postsyn_loc0 in enumerate(post_locs):
        for j, postsyn_loc1 in enumerate(post_locs):
            if j <= i:
                continue
            if not within_dist(postsyn_loc0, postsyn_loc1, threshold, min_threshold, scale):
                continue
            if not within_dist(pre_locs[i], pre_locs[j], threshold, min_threshold, scale):
                continue
            interpolated_points = generate_interpolates(postsyn_loc0, postsyn_loc1, n=10)
            sids = set([get_segment_id(seg, loc) for loc in interpolated_points])
            neuron_ids = [neuron_db.find_neuron_with_segment_id(sid, parent=True) for sid in sids]
            if len(set(neuron_ids)) == 1:
                nears.append((postsyn_loc0, postsyn_loc1))
    return nears

def merge_sets(lsts):
    # from https://stackoverflow.com/questions/9110837/python-simple-list-merging-based-on-intersections
    sets = [set(lst) for lst in lsts if lst]
    merged = True
    while merged:
        merged = False
        results = []
        while sets:
            common, rest = sets[0], sets[1:]
            sets = []
            for x in rest:
                if x.isdisjoint(common):
                    sets.append(x)
                else:
                    merged = True
                    common |= x
            results.append(common)
        sets = results
    return sets


def to_tuple(coord):
    return tuple([k for k in coord])

def sum_coords(coords):
    c = daisy.Coordinate((0, 0, 0))
    for cc in coords:
        c += cc
    return c

def average_syns(syn_list):
    new_syn = copy.deepcopy(syn_list[0])
    pres = [daisy.Coordinate(s['pre_loc']) for s in syn_list]
    syns = [daisy.Coordinate(s['syn_loc0']) for s in syn_list]
    posts = [daisy.Coordinate(s['post_loc']) for s in syn_list]
    avg_pre = sum_coords(pres)/len(pres)
    avg_syn = sum_coords(syns)/len(syns)
    avg_posts = sum_coords(posts)/len(posts)

    new_syn['syn_loc0'] = avg_syn
    new_syn['pre_loc'] = avg_pre
    new_syn['post_loc'] = avg_posts

    # for synapse length, we'll take the max
    new_syn['major_axis_length'] = max([s['major_axis_length'] for s in syn_list])

    # # for z length, we'll add the difference of top and bottom sections, + half of each
    # top_syn = sorted(syn_list, key=lambda s: s['post_loc'][2])[-1]
    # bottom_syn = sorted(syn_list, key=lambda s: s['post_loc'][2])[0]
    # new_z_len = abs(top_syn['post_loc'][2] - bottom_syn['pre_loc'][2])
    # new_z_len += int((top_syn['z_length'] + bottom_syn['z_length']) / 2)
    # new_syn['z_length'] = new_z_len
    # note2: let's just add z lengths together--more accurate
    new_syn['z_length'] = sum([s['z_length'] for s in syn_list])

    # print(syn_list)
    # print(f'merged: {new_syn}')

    # return tuple((to_tuple(avg_pre), to_tuple(avg_syn), to_tuple(avg_posts)))
    return new_syn

'''Load data'''

in_fname = sys.argv[1]
in_fname_base = in_fname.rsplit('.', 1)[0]

syndb = compress_pickle.load(in_fname)

# debug = True
debug = False

coalesced_db = defaultdict(lambda: defaultdict(list))

for grc, grc_syns in syndb.items():
    # print(grc)
    for pc, syns in grc_syns.items():
        pre_post_locs = [(s['pre_loc'], s['post_loc']) for s in syns]
        post_to_syn = dict()
        for s in syns:
            post_to_syn[s['post_loc']] = s
        nears = get_near_locs(pre_post_locs, seg, threshold=200)
        if debug and len(nears):
            print(f'Original pairs: {nears}')

        # need to average the position of merged synapses
        processed = set()
        combined_locss = merge_sets(nears)
        for combined_locs in combined_locss:
            for l in combined_locs:
                processed.add(l)
            combined_syns = [post_to_syn[loc] for loc in combined_locs]
            if debug:
                print(f'Merged pairs: {[to_ng(loc) for loc in combined_locs]}')
            combined_syn = average_syns(combined_syns)
            coalesced_db[grc][pc].append(combined_syn)

        for syn in syns:
            post_loc = syn['post_loc']
            if post_loc in processed:
                continue
            coalesced_db[grc][pc].append(syn)

fout = f'{in_fname_base}_coalesced.gz'

compress_pickle.dump((
    dict(coalesced_db)
    ), fout)


