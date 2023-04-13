import numpy as np
import h5py
# import daisy

from synful import synapse


def write_synapses_into_cremiformat(synapses, filename, offset=None,
                                    overwrite=False):
    id_nr, ids, locations, partners, types = 0, [], [], [], []
    for synapse in synapses:
        types.extend(['presynaptic_site', 'postsynaptic_site'])
        ids.extend([id_nr, id_nr + 1])
        partners.extend([np.array((id_nr, id_nr + 1))])
        assert synapse.location_pre is not None and synapse.location_post is not None
        locations.extend(
            [np.array(synapse.location_pre), np.array(synapse.location_post)])
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


outputfile = 'nosyn_cutout.hdf'
# Dummy fix. Currently, writing does not allow empyt list. This synapse it outside of training ROI.
synapses = [synapse.Synapse(location_pre=(0, 0, 0), location_post=(0, 0, 0))]

write_synapses_into_cremiformat(synapses, outputfile, overwrite=True)
