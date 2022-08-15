import random
import numpy as np
from numpy.random import default_rng
from neurons import Simulation
from collections import defaultdict

class RandomModel():

    def __init__(self, num_patterns, pattern_len, seed):
        random.seed(seed)
        rng = default_rng(seed)
        rng_integers = rng.integers

        patterns = []

        for i in range(num_patterns):
            v = rng_integers(0, 2, pattern_len, dtype=np.uint8)
            patterns.append((v, random.randint(0, 1)))

        self.patterns = patterns

    def get(self, batch_size, seed):
        random.seed(seed)
        ret = random.choices(self.patterns, k=batch_size)
        return {
            'pattern': np.array([r[0] for r in ret], dtype=np.float32),
            'cls': np.array([[r[1]] for r in ret], dtype=np.float32),
        }

    def get_dtype(self):
        return np.float32


class MfGcModel():

    def __init__(self, sim, num_patterns, variability,
                 test_noise, seed):
        random.seed(seed)

        self.test_noise = test_noise

        # calibrate model
        print("Calibrating model...")
        calibration_patterns = sim.generate_patterns(count=1024*4)
        sim.set_failure_rate(0, seed=seed)
        sim.evaluate(calibration_patterns, no_random=True,
                     calibrate_activation_level=0.3)

        # make random patterns with defined similarity level
        test_pattern = [calibration_patterns[0]]
        assert variability <= 1.0 and variability >= 0
        self.patterns = sim.add_noise_patterns(
            test_pattern, prob=variability, n=num_patterns,
            seed=seed, scaled_noise=0, same_output=False)
        self.sim = sim

    def get(self, batch_size, seed):
        random.seed(seed)
        # print(seed)
        redundant_patterns = random.choices(self.patterns, k=batch_size)

        # add noise to patterns
        if self.test_noise > 0.001:
            redundant_patterns = self.sim.add_noise_patterns(
                redundant_patterns, prob=self.test_noise, n=1,
                seed=None, scaled_noise=0, same_output=True)

        self.sim.evaluate(redundant_patterns, no_random=True)
        grc_acts = self.sim.get_grc_activities()

        return {
            'pattern': np.array(grc_acts, dtype=np.float32),
            'cls': np.array([[r[1]] for r in redundant_patterns], dtype=np.float32),
        }

    def get_dtype(self):
        return np.float32


class MfGcModelPartial():

    def __init__(self, sim, num_patterns, variability,
                 keep_pct,
                 test_noise, seed):
        random.seed(seed)

        self.test_noise = test_noise
        self.keep_pct = keep_pct
        assert keep_pct >= 0 and keep_pct <= 1

        # calibrate model
        print("Calibrating model...")
        calibration_patterns = sim.generate_patterns(count=1024*4)
        sim.set_failure_rate(0, seed=seed)
        sim.evaluate(calibration_patterns, no_random=True,
                     calibrate_activation_level=0.3)

        # make random patterns with defined similarity level
        test_pattern = [calibration_patterns[0]]
        assert variability <= 1.0 and variability >= 0
        self.patterns = sim.add_noise_patterns(
            test_pattern, prob=variability, n=num_patterns,
            seed=seed, scaled_noise=0, same_output=False)
        self.sim = sim

        # drop random # of outputs based on keep_pct
        self.keep_idx = None
        if keep_pct < 1.0:
            keep_grcs = int(sim.num_grcs*keep_pct)
            self.keep_idx = sorted(random.sample(range(sim.num_grcs), keep_grcs))

    def get(self, batch_size, seed):
        random.seed(seed)
        # print(seed)
        redundant_patterns = random.choices(self.patterns, k=batch_size)

        # add noise to patterns
        if self.test_noise > 0.001:
            redundant_patterns = self.sim.add_noise_patterns(
                redundant_patterns, prob=self.test_noise, n=1,
                seed=None, scaled_noise=0, same_output=True)

        self.sim.evaluate(redundant_patterns, no_random=True)
        grc_acts = self.sim.get_grc_activities()

        if self.keep_idx:
            grc_acts_ = []
            for i, pv in enumerate(grc_acts):
                grc_acts_.append(pv[self.keep_idx])
            grc_acts = grc_acts_

        return {
            'pattern': np.array(grc_acts, dtype=np.float32),
            'cls': np.array([[r[1]] for r in redundant_patterns], dtype=np.float32),
        }

    def get_dtype(self):
        return np.float32


class MfGcModelPartialWithCalibration():

    def __init__(self, sim, num_patterns, variability,
                 keep_pct,
                 test_noise, seed):
        random.seed(seed)

        self.test_noise = test_noise
        self.keep_pct = keep_pct
        assert keep_pct >= 0 and keep_pct <= 1

        # calibrate model
        print("Calibrating model...")
        calibration_patterns = sim.generate_patterns(count=1024*4)
        sim.set_failure_rate(0, seed=seed)
        sim.evaluate(calibration_patterns, no_random=True,
                     calibrate_activation_level=0.3)

        # make random patterns with defined similarity level
        test_pattern = [calibration_patterns[0]]
        assert variability <= 1.0 and variability >= 0
        self.patterns = sim.add_noise_patterns(
            test_pattern, prob=variability, n=num_patterns,
            seed=seed, scaled_noise=0, same_output=False)
        self.sim = sim

        # estimate output toggles wrt the output to filter out noise
        sim.evaluate(self.patterns, no_random=True)
        grc_acts = self.sim.get_grc_activities()
        patterns0 = [p for p, v in zip(grc_acts, self.patterns)
                     if v[1] == 0]
        patterns1 = [p for p, v in zip(grc_acts, self.patterns)
                     if v[1] == 1]
        # sample 2048 combinations and build histogram of SNR
        hist = defaultdict(int)
        for i in range(4096):
            p0 = random.choice(patterns0)
            p1 = random.choice(patterns1)
            for j, (m, n) in enumerate(zip(p0, p1)):
                if m != n:
                    hist[j] += 1

        hist_list = [(k, v) for k, v in hist.items()]
        hist_list.sort(key=lambda x: x[1], reverse=True)

        keep_grcs = int(sim.num_grcs*keep_pct)
        self.keep_idx = [k for k, v in hist_list][0:keep_grcs]

    def get(self, batch_size, seed):
        random.seed(seed)
        # print(seed)
        redundant_patterns = random.choices(self.patterns, k=batch_size)

        # add noise to patterns
        if self.test_noise > 0.001:
            redundant_patterns = self.sim.add_noise_patterns(
                redundant_patterns, prob=self.test_noise, n=1,
                seed=None, scaled_noise=0, same_output=True)

        self.sim.evaluate(redundant_patterns, no_random=True)
        grc_acts = self.sim.get_grc_activities()

        if self.keep_idx:
            grc_acts_ = []
            for i, pv in enumerate(grc_acts):
                grc_acts_.append(pv[self.keep_idx])
            grc_acts = grc_acts_

        return {
            'pattern': np.array(grc_acts, dtype=np.float32),
            'cls': np.array([[r[1]] for r in redundant_patterns], dtype=np.float32),
        }

    def get_dtype(self):
        return np.float32



class MfGcModelPartialWithCalibration2():

    def __init__(self, sim, num_patterns, variability,
                 keep_pct,
                 test_noise, seed):
        random.seed(seed)

        self.test_noise = test_noise
        self.keep_pct = keep_pct
        assert keep_pct >= 0 and keep_pct <= 1

        # calibrate model
        print("Calibrating model...")
        calibration_patterns = sim.generate_patterns(count=1024*4)
        sim.set_failure_rate(0, seed=seed)
        sim.evaluate(calibration_patterns, no_random=True,
                     calibrate_activation_level=0.3)

        # make random patterns with defined similarity level
        test_pattern = [calibration_patterns[0]]
        assert variability <= 1.0 and variability >= 0
        self.patterns = sim.add_noise_patterns(
            test_pattern, prob=variability, n=num_patterns,
            seed=seed, scaled_noise=0, same_output=False)
        self.sim = sim

        # estimate output toggles wrt the output to filter out noise
        sim.evaluate(self.patterns, no_random=True)
        grc_acts = self.sim.get_grc_activities()
        patterns0 = [p for p, v in zip(grc_acts, self.patterns)
                     if v[1] == 0]
        patterns1 = [p for p, v in zip(grc_acts, self.patterns)
                     if v[1] == 1]
        # sample 2048 combinations and build histogram of SNR
        hist = defaultdict(int)
        for i in range(4096):
            p0 = random.choice(patterns0)
            p1 = random.choice(patterns1)
            for j, (b0, b1) in enumerate(zip(p0, p1)):
                if b1 > b0:
                    hist[j] += 1

        hist_list = [(k, v) for k, v in hist.items()]
        hist_list.sort(key=lambda x: x[1], reverse=True)

        keep_grcs = int(sim.num_grcs*keep_pct)
        self.keep_idx = [k for k, v in hist_list][0:keep_grcs]

    def get(self, batch_size, seed):
        random.seed(seed)
        # print(seed)
        redundant_patterns = random.choices(self.patterns, k=batch_size)

        # add noise to patterns
        if self.test_noise > 0.001:
            redundant_patterns = self.sim.add_noise_patterns(
                redundant_patterns, prob=self.test_noise, n=1,
                seed=None, scaled_noise=0, same_output=True)

        self.sim.evaluate(redundant_patterns, no_random=True)
        grc_acts = self.sim.get_grc_activities()

        if self.keep_idx:
            grc_acts_ = []
            for i, pv in enumerate(grc_acts):
                grc_acts_.append(pv[self.keep_idx])
            grc_acts = grc_acts_

        return {
            'pattern': np.array(grc_acts, dtype=np.float32),
            'cls': np.array([[r[1]] for r in redundant_patterns], dtype=np.float32),
        }

    def get_dtype(self):
        return np.float32

