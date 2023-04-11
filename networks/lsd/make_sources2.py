import os

from gunpowder import ArrayKey, ZarrSource, ArraySpec, Pad, RandomLocation, \
                    DownSample, Normalize, Coordinate, Reject

from reject import Reject

dense_data_path= '../../data/dense/'
background_data_path= '../../data/background/'
negative_data_path = ../'../data/negative/'
sparse_data_path = ../'../data/sparse/'
special_min_masks = {}

dense_samples = [
    dense_data_path + 'gl0_setup45.zarr',
    dense_data_path + 'gl1_setup50_v2_large.zarr',
    dense_data_path + 'ml0_setup45.zarr',
    dense_data_path + 'ml4_setup45.zarr',
    dense_data_path + 'ml5_setup45.zarr',
    dense_data_path + 'pl2_setup45.zarr',
]

# myelin samples only have myelin set to background
# other neurons are also labeled but not proofread
# necessary that they are labeled so that the network is not biased toward background labeling for other cells
background_samples = [
    background_data_path + 'cb2_myelin_cutout0_v2.zarr',
    background_data_path + 'cb2_myelin_cutout2_v2.zarr',
    background_data_path + 'cb2_myelin_cube_v2.zarr',
]

# SPARSE SAMPLES
# only a few neurons are labeled and proofread
# other neurons are set to 0 in unlabeled_mask
sparse_samples = [
    sparse_data_path + 'ml_branch_boundaries_cropped.zarr',  # mito
    sparse_data_path + 'mcp1.zarr',  # mito
]
# these ones has very sparse labeling so we decrease the min mask to make it work
special_min_masks[sparse_data_path + 'ml_branch_boundaries_cropped.zarr'] = 0.2
special_min_masks[sparse_data_path + 'mcp1.zarr'] = 0.2


# NEGATIVE SAMPLES
# usually dense labeled with all cells proofread
negative_samples = []
negative_samples = [
    negative_data_path + 'cb2_negative_cutout1.zarr',
    negative_data_path + 'cb2_negative_cutout2.zarr',
    negative_data_path + 'cb2_negative_cutout3.zarr',
    negative_data_path + 'cb2_negative_cutout4.zarr',
    ]

new_samples = []
xy_downsample = 2

raw_fr = ArrayKey('RAW_FR')
labels_fr = ArrayKey('GT_LABELS_FR')
labels_mask_fr = ArrayKey('GT_LABELS_MASK_FR')
unlabeled_mask_fr = ArrayKey('GT_UNLABELED_MASK_FR')

if xy_downsample > 1:

    raw = ArrayKey('RAW')
    labels = ArrayKey('GT_LABELS')
    labels_mask = ArrayKey('GT_LABELS_MASK')
    unlabeled_mask = ArrayKey('GT_UNLABELED_MASK')

else:
    raw = ArrayKey('RAW_FR')
    labels = ArrayKey('GT_LABELS_FR')
    labels_mask = ArrayKey('GT_LABELS_MASK_FR')
    unlabeled_mask = ArrayKey('GT_UNLABELED_MASK_FR')


### CREATING GT WEIGHTS
total_samples = []
total_sample_weights = []
sample_mask_attrs = {}

def add_weights(samples, weight):
    global total_samples
    global total_sample_weights
    global sample_mask_attrs
    global special_min_masks

    global background_samples
    global sparse_samples
    global negative_samples
    global dense_samples

    for s in samples:

        assert os.path.exists(s), "Sample %s does not exist!" % s

        logical_and_mask = None
        logical_and_mask_val = 0
        mask = unlabeled_mask
        min_mask = 0.5

        if s in background_samples:
            mask = unlabeled_mask
            logical_and_mask = labels
            min_mask = 0.6

        if s in sparse_samples:
            assert s in special_min_masks
        if s in special_min_masks:
            min_mask = special_min_masks[s]

        total_samples.append(s)
        total_sample_weights.append(weight)
        sample_mask_attrs[s] = (
            mask, min_mask, logical_and_mask, logical_and_mask_val)

add_weights(dense_samples, 1)
add_weights(background_samples, 0.05)  # lower?
add_weights(negative_samples, 0.025)  # lower?
add_weights(sparse_samples, 0.075)  # 0.05 has bad performance at 100k, ok-ish at 200k
add_weights(new_samples, 5)


def create_source(sample):

    src = ZarrSource(
            sample,
            datasets={
                raw_fr: 'volumes/raw',
                labels_fr: 'volumes/labels/neuron_ids',
                labels_mask_fr: 'volumes/labels/labels_mask2',
                unlabeled_mask_fr: 'volumes/labels/unlabeled',
            },
            array_specs={
                raw_fr: ArraySpec(interpolatable=True),
                labels_fr: ArraySpec(interpolatable=False),
                labels_mask_fr: ArraySpec(interpolatable=False),
                unlabeled_mask_fr: ArraySpec(interpolatable=False),
            }
        )

    src += Pad(raw_fr, None)
    src += Pad(labels_fr, Coordinate((1000, 256, 256)))
    src += Pad(labels_mask_fr, Coordinate((1000, 256, 256)))
    src += Pad(unlabeled_mask_fr, Coordinate((1000, 256, 256)))
    src += RandomLocation()

    if xy_downsample > 1:
        src += DownSample(unlabeled_mask_fr, (1, xy_downsample, xy_downsample), unlabeled_mask)
        src += DownSample(labels_fr, (1, xy_downsample, xy_downsample), labels)

    mask, min_mask, logical_and_mask, logical_and_mask_val = sample_mask_attrs[sample]

    src += Reject(
        mask=mask,
        min_masked=min_mask,
        reject_probability=0.99,
        logical_and_mask=logical_and_mask,
        logical_and_mask_val=logical_and_mask_val,
        )

    if xy_downsample > 1:
        src += DownSample(raw_fr, (1, xy_downsample, xy_downsample), raw)
        src += DownSample(labels_mask_fr, (1, xy_downsample, xy_downsample), labels_mask)

    src += Normalize(raw)

    return src


def make_sources():

    data_sources = tuple(
        create_source(sample) for sample in total_samples
    )
    return data_sources
