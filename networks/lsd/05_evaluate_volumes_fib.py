#supress tensorflow future warnings lol
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# import malis
# import scipy
import daisy
import json
import logging
import lsd
import numpy as np
import os
import sys
import waterz
from funlib.evaluate import rand_voi
from funlib.segment.arrays import replace_values
from pymongo import MongoClient


logging.basicConfig(level=logging.INFO)

def evaluate(
        experiment,
        setup,
        iteration,
        gt_file,
        gt_dataset,
        fragments_file,
        fragments_dataset,
        db_host,
        rag_db_name,
        edges_collection,
        scores_db_name,
        thresholds_minmax,
        thresholds_step,
        volume,
        num_workers,
        configuration,
        roi_offset=None,
        roi_shape=None):

    # open fragments

    logging.info("Reading fragments from %s" %fragments_file)

    try:
        fragments = daisy.open_ds(fragments_file, fragments_dataset)
    except:
        fragments = daisy.open_ds(fragments_file, fragments_dataset + '/s0')

    logging.info("Reading gt from %s" %gt_file)

    try:
        gt = daisy.open_ds(gt_file, gt_dataset)
    except:
        gt = daisy.open_ds(gt_file, gt_dataset + '/s0')

    print('fragments: ', fragments)
    print('gt: ', gt)

    if roi_offset:
        common_roi = daisy.Roi(roi_offset, roi_shape)

    else:
        common_roi = fragments.roi.intersect(gt.roi)

    # evaluate only where we have both fragments and GT
    logging.info("Cropping fragments and GT to common ROI %s", common_roi)
    fragments = fragments[common_roi]
    gt = gt[common_roi]

    logging.info("Converting fragments to nd array...")
    fragments = fragments.to_ndarray()

    logging.info("Converting gt to nd array...")
    gt = gt.to_ndarray()

    thresholds = list(np.arange(
        thresholds_minmax[0],
        thresholds_minmax[1],
        thresholds_step))

    logging.info("Evaluating thresholds...")
    for threshold in thresholds:

        segment_ids = get_segmentation(
                fragments,
                fragments_file,
                edges_collection,
                threshold,
                volume)

        evaluate_threshold(
                experiment,
                setup,
                iteration,
                db_host,
                scores_db_name,
                edges_collection,
                segment_ids,
                gt,
                threshold,
                configuration,
                volume)

def get_segmentation(
        fragments,
        fragments_file,
        edges_collection,
        threshold,
        volume):

    logging.info("Loading fragment - segment lookup table for threshold %s..." %threshold)

    fragment_segment_lut_dir = os.path.join(
            fragments_file,
            'luts',
            'fragment_segment')

    # if volume:
        # logger.info("Run type set, evaluating on %s dataset", volume)

        # fragment_segment_lut_dir = os.path.join(
                    # fragment_segment_lut_dir,
                    # volume)

    fragment_segment_lut_file = os.path.join(
            fragment_segment_lut_dir,
            'seg_%s_%d.npz' % (edges_collection, int(threshold*100)))

    fragment_segment_lut = np.load(
            fragment_segment_lut_file)['fragment_segment_lut']

    assert fragment_segment_lut.dtype == np.uint64

    logging.info("Relabeling fragment ids with segment ids...")

    segment_ids = replace_values(fragments, fragment_segment_lut[0], fragment_segment_lut[1])

    return segment_ids

def evaluate_threshold(
        experiment,
        setup,
        iteration,
        db_host,
        scores_db_name,
        edges_collection,
        segment_ids,
        gt,
        threshold,
        configuration,
        volume):

        #open score DB
        client = MongoClient(db_host)
        database = client[scores_db_name]
        score_collection = database['scores']

        #get VOI and RAND
        logging.info("Calculating VOI scores for threshold %f...", threshold)

        logging.info(type(segment_ids))

        rand_voi_report = rand_voi(
                gt,
                segment_ids,
                return_cluster_scores=False)

        metrics = rand_voi_report.copy()

        for k in {'voi_split_i', 'voi_merge_j'}:
            del metrics[k]

        logging.info("Storing VOI values for threshold %f in DB" %threshold)

        metrics['threshold'] = threshold
        metrics['experiment'] = experiment
        metrics['setup'] = setup
        metrics['iteration'] = iteration
        metrics['method'] = configuration
        # metrics['run_type'] = volume[:len(volume)-5]
        metrics['run_type'] = volume
        metrics['merge_function'] = edges_collection.strip('edges_')

        print('VOI split: ', metrics['voi_split'])
        print('VOI merge: ', metrics['voi_merge'])

        logging.info(metrics)

        score_collection.replace_one(
                filter={
                    'method': metrics['method'],
                    'run_type': metrics['run_type'],
                    'merge_function': metrics['merge_function'],
                    'threshold': metrics['threshold']
                },
                replacement=metrics,
                upsert=True)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate(**config)
