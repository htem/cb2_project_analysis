import collections
import json
import sys


if __name__ == "__main__":

    try:
        folder = sys.argv[1]
    except:
        folder = '.'

    tree_json = json.load(open(folder + "/tree_geometry.json", 'r'))
    # print(tree_json['skeletons']['1163'].keys())

    pos_db = {}
    synapse_db = collections.defaultdict(lambda: collections.defaultdict(int))

    for skeleton in tree_json['skeletons']:
        for treenode in tree_json['skeletons'][skeleton]["treenodes"]:
            loc = tree_json['skeletons'][skeleton]["treenodes"][treenode]["location"]
            # print(loc)
            pos_db[int(treenode)] = (
                float(loc[2]),
                float(loc[1]),
                float(loc[0]),
                )

        for synapse_id in tree_json['skeletons'][skeleton]["connectors"]:
            synapse = tree_json['skeletons'][skeleton]["connectors"][synapse_id]

            for synapse_type in ["presynaptic_to", "postsynaptic_to"]:

                if len(synapse[synapse_type]):
                    assert synapse_db[synapse_id][synapse_type] == 0
                    assert len(synapse[synapse_type]) == 1
                    synapse_db[synapse_id][synapse_type] = synapse[synapse_type][0]

    # print(synapse_db)
    out = folder + "/ground_truth.csv"
    n = 0
    with open(out, 'w') as file:
        file.write('pre_z,pre_y,pre_x,post_z,post_y,post_x\n')
        for synapse in synapse_db:
            synapse = synapse_db[synapse]
            # print(synapse)
            if 'presynaptic_to' not in synapse:
                continue
            if 'postsynaptic_to' not in synapse:
                continue
            n += 1
            pre_zyx = pos_db[synapse['presynaptic_to']]
            post_zyx = pos_db[synapse['postsynaptic_to']]
            file.write(
                str(pre_zyx[0]) + ',' +
                str(pre_zyx[1]) + ',' +
                str(pre_zyx[2]) + ',' +
                str(post_zyx[0]) + ',' +
                str(post_zyx[1]) + ',' +
                str(post_zyx[2]) + '\n'
                )

    print("Num synapses: %d" % n)
