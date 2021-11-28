import random

from neurons import GranuleCell, MossyFiber, Simulation


if __name__ == '__main__':

    random.seed(0)

    sim = Simulation(
        num_grc=1000,
        num_mf=200,
        num_dendrite=7,
        grc_act_threshold=5,
        )
    n_pattern = 32
    patterns = sim.generate_patterns(count=n_pattern)
    # sim.print_grc_weights()

    sim.reset()
    sim.grc_act_threshold = .999
    sim.train(patterns)
    sim.evaluate(patterns)

    # sim.reset()
    # sim.grc_act_threshold = .999
    # sim.train(patterns)
    # sim.evaluate(patterns)

    # sim.reset()
    # sim.grc_act_threshold = .99
    # sim.train(patterns)
    # sim.evaluate(patterns)

    # sim.grc_act_threshold = .8
    # sim.train(patterns)
    # sim.evaluate(patterns)
    # sim.reset()
