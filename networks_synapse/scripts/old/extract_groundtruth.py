import collections
# import glob
# import itertools
# import os
import json

# import daisy
# import h5py
# import numpy as np
import sys

# from synful import database


if __name__ == "__main__":

    try:
        folder = sys.argv[1]
    except:
        folder = '.'

    tree_json = json.load(open(folder + "/tree_geometry.json", 'r'))
    # print(tree_json['skeletons']['1163'].keys())

    pos_db = {}
    for skeleton in tree_json['skeletons']:
        for treenode in tree_json['skeletons'][skeleton]["treenodes"]:
            loc = tree_json['skeletons'][skeleton]["treenodes"][treenode]["location"]
            # print(loc)
            pos_db[int(treenode)] = (
                float(loc[2]),
                float(loc[1]),
                float(loc[0]),
                )


    # pos_csv = folder + "/skeleton_coordinates.csv"
    # col_map = None
    # pos_db = {}
    # with open(pos_csv, 'r') as pos_file:
    #     for line in pos_file:
    #         if col_map is None:
    #             col_map = {}
    #             for i, f in enumerate(line.split(',')):
    #                 col_map[f.strip()] = i
    #             # print(col_map)
    #             continue

    #         f = line.split(',')
    #         pos_db[int(f[col_map['treenode_id']])] = (
    #             float(f[col_map['z']].strip()),
    #             float(f[col_map['y']].strip()),
    #             float(f[col_map['x']].strip()))

    # print(pos_db)

    synapse_csv = folder + "/connectors.csv"
    col_map = None
    synapse_db = collections.defaultdict(dict)
    with open(synapse_csv, 'r') as synapse_file:
        for line in synapse_file:
            if col_map is None:
                col_map = {}
                for i, f in enumerate(line.split(',')):
                    col_map[f.strip()] = i
                # print(col_map)
                continue

            f = line.split(',')
            # print(f[col_map['relation_id']])
            key = ('post' if f[col_map['relation_id']].strip() == "postsynaptic_to"
                   else 'pre')
            synapse_db[int(f[col_map['connector_id']])][key] = (
                int(f[col_map['treenode_id']].strip()))

    # print(synapse_db)
    out = folder + "/ground_truth.csv"
    with open(out, 'w') as file:
        file.write('pre_z,pre_y,pre_x,post_z,post_y,post_x\n')
        for synapse in synapse_db:
            synapse = synapse_db[synapse]
            print(synapse)
            pre_zyx = pos_db[synapse['pre']]
            post_zyx = pos_db[synapse['post']]
            file.write(
                str(pre_zyx[0]) + ',' +
                str(pre_zyx[1]) + ',' +
                str(pre_zyx[2]) + ',' +
                str(post_zyx[0]) + ',' +
                str(post_zyx[1]) + ',' +
                str(post_zyx[2]) + '\n'
                )
