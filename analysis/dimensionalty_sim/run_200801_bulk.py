import random
import sys
from multiprocessing import Pool
from functools import partial
import argparse
import copy

from neurons import GranuleCell, MossyFiber, Simulation


def run(
        num_dendrite,
        config,
        delta=0.025,
        input_noise=0.05,
        decoder_error_margin=.1,
        min_act_level=.5,
    ):

    random.seed(0)
    sim = Simulation(
        num_grc=config['num_grc'],
        num_mf=config['num_mf'],
        num_dendrite=num_dendrite,
        grc_act_threshold=5,
        default_input_noise=input_noise,
        default_decoder_error_margin=decoder_error_margin,
        grc_act_on_failure_rate=config['grc_act_on_failure_rate'],
        grc_act_off_failure_rate=config['grc_act_off_failure_rate'],
        min_train_it=config['min_train_it'],
        n_evaluate_sampling=config['n_evaluate_sampling'],
        evaluate_sampling_majority=config['evaluate_sampling_majority'],
        max_synapse_weight=config['max_synapse_weight'],
        # grc_mf_ratio=config['grc_mf_ratio'],
        )

    patterns = sim.generate_patterns(count=config['n_pattern'])
    # mult patterns
    patterns_0 = copy.deepcopy(patterns)
    for i in range(config['pattern_mult']-1):
        ps = []
        for j in range(len(patterns_0)):
            p = sim.add_input_noise(patterns_0[j][0], config['pattern_mult_noise'])
            # output = patterns_0[j][1]
            # if config['pattern_mult_opposite_output']:
            #     output = ~output
            # ps.append((p, output))
            if config['pattern_mult_opposite_output']:
                ps.append((p, not patterns_0[j][1]))
            else:
                ps.append((p, patterns_0[j][1]))
        patterns.extend(ps)

    results = []
    thresholds = []
    # for i in range(int(0.05/delta)):
    for i in range(int((1.0-min_act_level)/delta)):
        thresholds.append(min_act_level + i * delta)
    thresholds.append(.999)

    # print(thresholds)
    if config['test_random_pattern']:
        test_patterns = sim.generate_patterns(count=config['n_pattern'])
    else:
        test_patterns = patterns

    if config['pattern_mult_test_only']:
        train_patterns = patterns_0
    else:
        train_patterns = patterns

    for threshold in thresholds:
        sim.reset()
        sim.grc_act_threshold = threshold
        sim.train(train_patterns)
        # sim.print_grc_weights()
        results.append(sim.evaluate(test_patterns))

    print(f'Results for {num_dendrite}: {results}')
    return results


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--num_grc", type=int, default=1000)
    ap.add_argument("--num_mf", type=int, default=200)
    ap.add_argument("--n_pattern", type=int, default=32)
    ap.add_argument("--pattern_mult", type=int, default=1)
    ap.add_argument("--pattern_mult_noise", type=float, default=0.1)
    ap.add_argument("--pattern_mult_test_only", type=bool, default=False)
    ap.add_argument("--pattern_mult_opposite_output", type=bool, default=False)
    ap.add_argument("--grc_act_on_failure_rate", type=float, default=0.0)
    ap.add_argument("--grc_act_off_failure_rate", type=float, default=0.0)
    ap.add_argument("--n_evaluate_sampling", type=int, default=1)
    ap.add_argument("--evaluate_sampling_majority", type=bool, default=False)
    ap.add_argument("--min_train_it", type=int, default=15000)
    ap.add_argument("--max_synapse_weight", type=int, default=255)
    ap.add_argument("--test_random_pattern", type=bool, default=False)
    # ap.add_argument("grc_mf_ratio", type=float)

    config = vars(ap.parse_args())
    random.seed(0)

    n_dendrites = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    p = Pool(len(n_dendrites))
    fn = partial(
        run,
        input_noise=0.1,
        # min_act_level=.95,
        min_act_level=.8,
        config=config,
        )

    p_results = p.map(fn, n_dendrites)

    for l in p_results:
        for ll in l:
            print(ll, end=', ')
        print()
