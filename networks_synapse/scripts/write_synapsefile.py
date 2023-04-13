import numpy as np
import h5py
import sys
# import daisy

from synful import synapse


def write_synapses_into_cremiformat(synapses, filename, offset=None,
                                    overwrite=False):
    id_nr, ids, locations, partners, types = 0, [], [], [], []
    for syn in synapses:
        types.extend(['presynaptic_site', 'postsynaptic_site'])
        ids.extend([id_nr, id_nr + 1])
        partners.extend([np.array((id_nr, id_nr + 1))])
        assert syn.location_pre is not None and syn.location_post is not None
        locations.extend(
            [np.array(syn.location_pre), np.array(syn.location_post)])
        id_nr += 2
    if overwrite:
        h5_file = h5py.File(filename, 'w')
    else:
        h5_file = h5py.File(filename, 'a')
    dset = h5_file.create_dataset('annotations/ids', data=ids,
                                  compression='gzip')
    dset = h5_file.create_dataset('annotations/locations',
                                  data=np.stack(locations, axis=0).astype(
                                      np.float32),
                                  compression='gzip')
    dset = h5_file.create_dataset('annotations/presynaptic_site/partners',
                                  data=np.stack(partners, axis=0).astype(
                                      np.uint32),
                                  compression='gzip')
    dset = h5_file.create_dataset('annotations/types',
                                  data=np.array(types, dtype='S'),
                                  compression='gzip')

    if offset is not None:
        h5_file['annotations'].attrs['offset'] = offset
    h5_file.close()
    print('File written to {}'.format(filename))


inputfile = sys.argv[1]
outputfile = sys.argv[2]
synapses = []

for line in open(inputfile):
    nums = line.strip().split(',')
    if len(nums) != 6 or nums[0] == "pre_z":
        continue
    pre_z, pre_y, pre_x, post_z, post_y, post_x = nums
    syn = synapse.Synapse(
        location_pre=(pre_z, pre_y, pre_x),
        location_post=(post_z, post_y, post_x),
        )
    synapses.append(syn)

print("len(synapses) ", len(synapses))

write_synapses_into_cremiformat(synapses, outputfile, overwrite=True)
