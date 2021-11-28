import argparse
import random
import copy
import logging
import sys
import os

from random_patterns import generate_patterns, add_noise_to_patterns

from run_tests_210617 import *
import make_graph_210618
from make_graph_210618 import make_graph



def run_config(config, script_n):

    if config.weight_type == 'same':
        make_weights_fn = functools.partial(get_optimal_weights_same,
                                            valence_dir=config.valence_dir,
                                            )
    elif config.weight_type == 'change':
        make_weights_fn = functools.partial(get_optimal_weights_change,
                                            valence_dir=config.valence_dir,
                                            irrelevant_bits=config.irrelevant_bits,
                                            )
    else:
        assert False

    n_grcs = config.n_grcs
    n_mfs = config.n_mfs
    model = config.model
    assert model is not None

    pattern_type = config.pattern_type
    pattern_generator = functools.partial(generate_patterns, type=pattern_type)
    # noise_generator = functools.partial(add_noise_to_patterns, type=pattern_type, small_feature_mode=True)
    variation_generator = functools.partial(add_noise_to_patterns, type=pattern_type)
    noise_generator = functools.partial(add_noise_to_patterns, type=pattern_type,
        invert_noise_mask=config.invert_noise_mask)

    def test(model, seed):
        graph, model_desc = make_graph(model, seed=seed, n_grcs=n_grcs, n_mfs=n_mfs, config=config)
        sim = make_sim(graph)
        if pattern_type == 'binary':
            sim.set_binary_mode()
            pass
        if config.top_mf_mask is not None:
            # print(make_graph_210618.count_redundancy(sim)); asdf
            top_mf_mask = make_graph_210618.get_top_mf_mask(sim, config.top_mf_mask)
            variation_generator = functools.partial(add_noise_to_patterns, type=pattern_type, mf_mask=top_mf_mask)

        return model_desc, test_consistency_across_variations(
            sim,
            pattern_generator=pattern_generator,
            variation_generator=variation_generator,
            noise_generator=noise_generator,
            print_output=True,
            test_len=config.pattern_len,
            activation_level=config.activation_level,
            variation_sizes=config.variation_sizes,
            noise_level=config.noise_level,
            make_weights_fn=make_weights_fn,
            seed=seed,
            )

    print(f'Running {model}')
    ress = []
    for n in range(config.n_random):
        if n == 37:
            continue # bug with pass 37?
        print(f'Pass {n}')
        model_desc, res = test(model, seed=n)
        ress.append(res)
        assert model_desc is not None
        compress_pickle.dump((
            ress,
            ), f"{script_n}/{script_n}_{model_desc}_"
               f"{pattern_type}_{config.n_grcs}_{config.n_mfs}_"
               f"dir_{config.valence_dir}_noise_{config.noise_level}_"
               f"{config.activation_level}_{config.pattern_len}_{config.n_random}.gz")
        print()

