import argparse
import random
import copy
import logging
import sys
import os

from random_patterns2 import generate_patterns, add_noise_to_patterns, add_noise_patterns_using_masks
from global_random_model import GlobalRandomModel

from run_tests_220720_snr import *
import make_graph_220721 as make_graph

def run_config(config, script_n, save_args="", test_fn=None,
        # variation_generator=None,
        random_variation_kwargs=None,
        seed=0,
        batch_size=1,
        ):

    n_grcs = config.n_grcs
    n_mfs = config.n_mfs
    model = config.model
    assert model is not None
    pattern_type = config.pattern_type
    if random_variation_kwargs is None:
        random_variation_kwargs = {}
    pattern_generator = functools.partial(generate_patterns, type=pattern_type)

    # if variation_generator is not None:
    #     default_variation_generator = variation_generator
    # else:
    # default_variation_generator = functools.partial(
    #         add_noise_patterns_using_masks, type=pattern_type,
    #         **random_variation_kwargs)

    default_variation_generator = functools.partial(
            add_noise_patterns_using_masks,
            pattern_type=pattern_type,
            **random_variation_kwargs)

    def test(model, seed):
        graph, model_desc = make_graph.make_graph(
            model, seed=seed, n_grcs=n_grcs, n_mfs=n_mfs, config=config)
        per_bouton = True
        if isinstance(graph, GlobalRandomModel):
            per_bouton = False
        sim = make_sim(graph, per_bouton=per_bouton)
        if pattern_type == 'binary':
            sim.set_binary_mode()
            pass

        # mf_mask = None
        # if mf_mask is not None:
        #     variation_generator = functools.partial(
        #         add_noise_to_patterns, type=pattern_type,
        #         mf_mask=mf_mask, no_adjust_noise_ratio=True,
        #         **random_variation_kwargs)
        # else:
        variation_generator = default_variation_generator

        if test_fn is not None:
            return model_desc, test_fn(
                sim,
                pattern_generator=pattern_generator,
                noise_generator=variation_generator,
                print_output=True,
                # test_len=config.pattern_len,
                activation_level=config.activation_level,
                # noise_probs=config.variation_sizes,
                seed=seed,
                )
        else:
            return model_desc, test_across_noise(
                sim,
                pattern_generator=pattern_generator,
                noise_generator=variation_generator,
                print_output=True,
                test_len=config.pattern_len,
                activation_level=config.activation_level,
                noise_probs=config.variation_sizes,
                seed=seed,
                )

    n = seed
    print(f'Running {model}')
    print(f'Pass {n}')
    # seed = n
    model_desc, res = test(model, seed=n)

    return model_desc, res

    # # ress.append(res)
    # assert model_desc is not None
    # compress_pickle.dump((
    #     res,
    #     ), f"{script_n}/{script_n}_{model_desc}_"
    #        f"{pattern_type}_{config.n_grcs}_{config.n_mfs}_"
    #        # f"dir_{config.valence_dir}_noise_{config.noise_level}_"
    #        f"{save_args}"
    #        f"{config.activation_level}_{config.pattern_len}_{n}.gz")
    # print()
