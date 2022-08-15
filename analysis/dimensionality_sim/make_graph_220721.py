import copy
from collections import defaultdict
import compress_pickle
import random

from global_random_model2 import GlobalRandomModel

import tools_mf_graph

from make_expanded_graph import grc_dendrite_dist, mf_size_dist

# input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/gen_db/mf_grc/input_graph_210611_grc_mf_limited_100_2_xlim_360000_600000.gz')
# input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/gen_db/mf_grc/input_graph_210611_grc_center_z_100_2_xlim_360000_600000_x_margin_20_z_margin_10.gz')

input_graph = None

expanded_graph_path = 'input_graph_210611_expanded_graph_220722.gz'
default_input_graph_path = 'input_graph_210611_grc_center_z_100_2_xlim_360000_600000_x_margin_20_z_margin_5.0.gz'
'''
observed sharing:
0: 1391.7628512679917
1: 62.74434544208362
2: 3.365318711446196
3: 0.12063056888279644
4: 0.006854009595613434

local_random sharing:
0: 1400.3591501028102
1: 55.18848526387937
2: 2.3783413296778617
3: 0.07128169979437972
4: 0.0027416038382453737
'''


def set_input_graph(input_graph_path):
    global input_graph
    global gt_claw_count_hist
    input_graph = compress_pickle.load(input_graph_path)

    # gt_claw_count_hist = []
    # for _, grc in input_graph.grcs.items():
    #     # gt_claw_count_hist[grc.num_claws_gt] += 1
    #     gt_claw_count_hist.append(grc.num_claws_gt)
    #     # gt_claw_count_hist = dict(gt_claw_count_hist)


def limit_mask(mask, mask_limit, mf_margin_mask=None):
    assert mask_limit >= 0 and mask_limit <= 1

    full_len = len(mask)
    if mf_margin_mask is not None:
        full_len = len(mf_margin_mask)
    valid_idx = []
    for k, v in enumerate(mask):
        if v:
            valid_idx.append(k)
    random.shuffle(valid_idx)
    ret = [0]*len(mask)
    for i in valid_idx[0:int(full_len*mask_limit)]:
        ret[i] = 1
    return ret


def get_biggest_mfs_mask(sim, top_pct, in_mask=None, mf_mask_limit=None):
    # print(sim.mf_size); asdf
    # print(sorted([k for j, k in sim.mf_size.items()]))
    # asdf
    mf_size = [[k, v] for k, v in sim.mf_size.items()]
    # mf_size.sort(key=lambda x: x[1])
    # print(len(sim.mfs))
    # print(len(mf_size)); asdf
    # print(mf_size)
    # for mf_id, size in mf_size:
    #     print(f'{sim.mfs[mf_id].locs[0]}: {size}')

    mf_size_filtered = mf_size
    if in_mask is not None:
        mf_size_filtered = [m for m in mf_size if in_mask[m[0]]]


    # if in_mask is not None:
    #     for item in mf_size:
    #         if in_mask[item[0]] == 0:
    #             item[1] = 0
    mf_size_filtered.sort(key=lambda x: x[1])

    # mf_size_debug = [v for k, v in mf_size]

    in_mask_pct = 1
    if in_mask is not None:
        in_mask_pct = sum(in_mask) / len(in_mask)
    assert in_mask_pct <= 1

    count = int(len(mf_size)*abs(top_pct)*in_mask_pct+.5)

    if top_pct < 0:
        biggest_mfs = [k for k, v in mf_size_filtered[0:count]]
    else:
        biggest_mfs = [k for k, v in mf_size_filtered[-count:]]

    mask = [0]*len(sim.mf_size)
    for i in sim.mf_size.keys():
        # mask.append(1 if i in biggest_mfs else 0)
        if i in biggest_mfs:
            mask[i] = 1
    # mask = [1 for i in mf_size if i in biggest_mfs else 0]
    return mask

def get_center_mf_mask(sim, z_margin):
    zlim = [70*40+z_margin, 1170*40-z_margin]
    print(f'zlim: {zlim}')
    assert zlim[1] >= zlim[0]
    mask = []
    for mf in sim.mfs:
        within_zlim = False
        for loc in mf.locs:
            if loc[2] > zlim[0] and loc[2] < zlim[1]:
                within_zlim = True
        if within_zlim:
            mask.append(1)
        else:
            mask.append(0)
    return mask

def make_graph(model, n_grcs, n_mfs,
            config,
            seed=0):

    global input_graph
    global grc_dendrite_dist
    global mf_size_dist
    if input_graph is None:
        set_input_graph(default_input_graph_path)

    if model == 'observed':
        g = input_graph
        model_desc = f'{model}'

    elif model == 'global_random':
        g = GlobalRandomModel(
            n_grcs=n_grcs,
            n_mfs=n_mfs,
            n_dendrites=grc_dendrite_dist,
            seed=seed,
            )
        model_desc = f'{model}'

    elif model == 'global_random_fixed_mf':
        g = GlobalRandomModel(
            n_grcs=n_grcs,
            n_mfs=n_mfs,
            n_dendrites=grc_dendrite_dist,
            mf_size_dist=mf_size_dist,
            seed=seed,
            )
        model_desc = f'{model}'

    elif model == 'local_random_mf_size':
        '''local random with randomized bouton size'''
        g = input_graph
        tools_mf_graph.randomize_mf_size_by_depth(
                g, z_bucket_size=2000, z_offset=2800, seed=seed)  # shuffle bouton size in depth buckets
        g.randomize_graph_by_grc2(
            # constant_dendrite_length=True,
            mf_dist_margin=5000,
            # approximate_mf_degree=True,
            preserve_mf_degree=True,
            seed=seed,
            )

        model_desc = f'{model}'

    elif model == 'local_random':
        # g = copy.deepcopy(input_graph)
        g = input_graph
        g.randomize_graph_by_grc2(
            constant_dendrite_length=True,
            mf_dist_margin=5000,
            # approximate_mf_degree=True,
            seed=seed,
            )

        model_desc = f'{model}'

    elif model == 'local_random_length':
        g = copy.deepcopy(input_graph)

        try:
            dendrite_length = config.dendrite_length
        except:
            dendrite_length = 15000

        g.randomize_graph_by_grc2(
            constant_dendrite_length=dendrite_length,
            mf_dist_margin=5000,
            # approximate_mf_degree=True,
            seed=seed,
            )

        model_desc = f'{model}_{dendrite_length}'

    elif model == 'local_random_expanded':

        g = compress_pickle.load(expanded_graph_path)
        tools_mf_graph.randomize_size(g, grc_dendrite_dist=g.grc_dendrite_dist, mf_size_dist=g.mf_size_dist, seed=seed)
        try:
            dendrite_length = config.dendrite_length
        except:
            dendrite_length = 21000
        g.randomize_graph_by_grc2(
            constant_dendrite_length=dendrite_length,
            mf_dist_margin=5000,
            single_connection_per_pair=1,
            seed=seed,
            )
        model_desc = f'{model}_length_{dendrite_length}'

    elif model == 'local_random_expanded_fixed_mf':

        g = compress_pickle.load(expanded_graph_path)
        tools_mf_graph.randomize_size(g, grc_dendrite_dist=g.grc_dendrite_dist, mf_size_dist=g.mf_size_dist, seed=seed)
        try:
            dendrite_length = config.dendrite_length
        except:
            dendrite_length = 21000
        g.randomize_graph_by_grc2(
            constant_dendrite_length=dendrite_length,
            mf_dist_margin=5000,
            single_connection_per_pair=1,
            approximate_mf_degree=True,
            seed=seed,
            )
        model_desc = f'{model}_length_{dendrite_length}'

    else:
        raise RuntimeError(f'model `{model}` not found!')

    tools_mf_graph.count_redundancy_graph(g)

    try:
        grc_dendrite_dist, mf_size_dist = tools_mf_graph.get_distributions(
            g, z_margin_pct=40, x_margin_pct=25)
        print("\ngrc_dendrites:")
        print(len(grc_dendrite_dist))
        tools_mf_graph.print_box_plot(grc_dendrite_dist)
        print("\nmf_size_dist:")
        print(len(mf_size_dist))
        tools_mf_graph.print_box_plot(mf_size_dist)
        print()
    except:
        pass

    if n_grcs != len(g.grcs) or n_mfs != len(g.mf_locs):
        print(f'Mismatched graph size: {n_grcs}/{n_mfs} vs {len(g.grcs)}/{len(g.mf_locs)}')
        assert False

    return g, model_desc


def count_redundancy(sim):
    # grcs_claws = []
    # mf_to_grcs = defaultdict(set)
    # for grc_id, dendrite_count in enumerate(g.dendrite_counts):
    #     claws = []
    #     for j in range(dendrite_count):
    #         mf_id = g.dendrite_mf_map[pos]
    #         pos += 1
    #         claws.append(mf_id)
    #         mf_to_grcs[mf_id].add(grc_id)
    #     grcs_claws.append(set(claws))
    # nshares = defaultdict(int)
    # for mf_id, grcs in mf_to_grcs.items():
    #     for pair in itertools.combinations(grcs, 2):
    #         nshare = len(grcs_claws[pair[0]] & grcs_claws[pair[1]])
    #         nshares[nshare] += 1
    # for n in sorted(nshares.keys()):
    #     print(f'{n}: {nshares[n]/len(g.dendrite_counts)}')
    nshares = defaultdict(int)
    for grc_id_i, grc_i in enumerate(sim.grcs):
        for grc_id_j, grc_j in enumerate(sim.grcs):
            if grc_id_i == grc_id_j:
                continue
            nshare = len(set(grc_i.claws) & set(grc_j.claws))
            nshares[nshare] += 1
    for n in sorted(nshares.keys()):
        print(f'{n}: {nshares[n]/len(sim.grcs)}')

