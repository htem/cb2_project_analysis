import random
import sys
from multiprocessing import Pool
from functools import partial

from neurons import GranuleCell, MossyFiber, Simulation


def run(
        num_dendrite,
        n_pattern,
        delta=0.025,
        input_noise=0.05,
        decoder_error_margin=.1,
        min_act_level=.5,
    ):

    random.seed(0)
    sim = Simulation(
        num_grc=1000,
        num_mf=200,
        num_dendrite=num_dendrite,
        grc_act_threshold=5,
        default_input_noise=input_noise,
        default_decoder_error_margin=decoder_error_margin,
        min_train_it=500,
        min_eval_it=2000,
        )
    patterns = sim.generate_patterns(count=n_pattern)

    results = []
    thresholds = []
    # for i in range(int(0.05/delta)):
    for i in range(int((1.0-min_act_level)/delta)):
        thresholds.append(min_act_level + i * delta)
    thresholds.append(.999)

    # print(thresholds)

    for threshold in thresholds:
        sim.reset()
        sim.grc_act_threshold = threshold
        sim.train(patterns)
        # sim.init_grcs()
        # sim.print_grc_weights()
        results.append(sim.evaluate(patterns, output_act_lv=True))

    print(f'Results for {num_dendrite}: {results}')
    return results



if __name__ == '__main__':

    random.seed(0)

    n_pattern = int(sys.argv[1])

    p = Pool(10)
    fn = partial(
        run, n_pattern=n_pattern, input_noise=0.1,
        # min_act_level=.95,
        min_act_level=.6,
        )

    n_dendrites = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # n_dendrites = [3, 4, 5]
    p_results = p.map(fn, n_dendrites)

    # print(zip(n_dendrites, p_results))
    # print(p_results)
    for l in p_results:
        for ll in l:
            print(ll, end=', ')
        print()


