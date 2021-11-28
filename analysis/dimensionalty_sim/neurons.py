import random
import math
import copy
import numpy as np
import logging
import collections


logger = logging.getLogger(__name__)


class GranuleCell():
    def __init__(
        self,
        # num_mfs,
        # num_dendrite,
        claws,
        claw_weights=None,
        # act_threshold,
        # grc_act_on_failure_rate,
        # grc_act_off_failure_rate,
        # max_weight=255,
        # act_lv=0.01,
    ):
        self.activations = []
        self.inputs = []
        # self.act_threshold = act_threshold
        # self.max_weight = max_weight
        # self.output_weight = int(max_weight/2)
        # self.grc_act_on_failure_rate = grc_act_on_failure_rate
        # self.grc_act_off_failure_rate = grc_act_off_failure_rate
        # if act_threshold < 1:
        #     act_threshold = act_threshold*num_dendrite
        # self.act_threshold = act_threshold
        self.claws = claws
        # while len(self.claws) < num_dendrite:
        #     mf_id = random.randint(0, num_mfs-1)
        #     if mf_id not in self.claws:
        #         self.claws.append(mf_id)
        #     self.claws.sort()
            # print(self.claws)
        self.claws.sort()
        if claw_weights:
            self.claw_weights = claw_weights
            assert False, "Untested"
        else:
            self.claw_weights = [1]*len(self.claws)
        self.activated = False
        self.act_lv_scale = 1
        self.broken = False
        # self.act_lv = act_lv

    def activate(
            self, pattern,
            # grc_act_off_failure_rate=None,
            ):
        if self.broken:
            self.activations.append(0)
            self.activated = False
            return False
        sum = 0.0
        for i, claw in enumerate(self.claws):
            sum += pattern[claw]
        self.inputs.append(sum)
        if sum >= self.act_lv_scale:
            activated = True
            self.activations.append(1)
        else:
            activated = False
            self.activations.append(0)
        self.activated = activated
        return activated

    def train(self, input_mfs, output):
        act = False
        act = self.activate(input_mfs, grc_act_off_failure_rate=0)
        if act:
            if output:
                self.output_weight = min(
                    self.output_weight+1, self.max_weight)
            else:
                self.output_weight = max(
                    self.output_weight-1, 0)
    def reset(self):
        self.inputs = []
        self.activations = []
    def calibrate_activation_level(self, act_lv):
        self.inputs.sort()
        idx = int((1-act_lv)*len(self.inputs))
        if self.inputs[idx] == 0:
            # print(self.inputs)
            print(self.claws)
        # scale = 1.0 / self.inputs[idx]
        # self.act_lv_scale = scale
        self.act_lv_scale = self.inputs[idx]


class MossyFiber():
    def __init__(self, mf_id):
        self.mf_id = mf_id
        self.activations = []
        pass
    def reset(self):
        self.activations = []
    def activate(self, pattern):
        self.activations.append(pattern[self.mf_id])


class Simulation():

    def __init__(
        self,
        input_graph,
        # num_grc=None,
        # num_mfs=None,
        # num_dendrite=None,
        # grc_act_threshold=None,
        # grc_act_on_failure_rate=0,
        # grc_act_off_failure_rate=0,
        # max_synapse_weight=255,
        # min_train_it=15000,
        min_eval_it=5000,
        # default_input_noise=0.05,
        # default_decoder_error_margin=0.10,
        # n_evaluate_sampling=1,
        # evaluate_sampling_majority=False,
    ):
        self.num_mfs = len(input_graph.mfs)
        self.num_grcs = len(input_graph.grcs)
        self.min_eval_it = min_eval_it
        self.init_mfs()
        self.init_grcs(input_graph)
        self.failure_rate = None

    def reset(self):
        for grc in self.grcs:
            grc.reset()
        for mf in self.mfs:
            mf.reset()
        # random.seed(0)

    def init_mfs(self):
        self.mfs = []
        for i in range(self.num_mfs):
            self.mfs.append(MossyFiber(mf_id=i))

    def init_grcs(self, input_graph):
        self.grcs = []
        mapping = {}
        counter = 0
        for mf_id, mf in input_graph.mfs.items():
            mapping[mf_id] = counter
            counter += 1
        for grc_id, grc in input_graph.grcs.items():
            claws = [mapping[mf_id] for mf_id, _ in grc.edges]
            self.grcs.append(
                GranuleCell(
                    claws=claws,
                    )
                )

    def set_failure_rate(self, failure_rate, seed):
        random.seed(seed)
        for grc in self.grcs:
            grc.broken = True if random.random() < failure_rate else False

    def generate_patterns(
            self,
            count,
            type='random',
            # independent_noise=0,
        ):
        patterns = []
        # outputs = []
        pattern_len = self.num_mfs

        for i in range(count):
            if type == 'random':
                b = [None]*pattern_len
                for k in range(pattern_len):
                    b[k] = random.random()
            elif type == 'gaussian':
                mu, sigma = 0.5, 0.2 # mean and standard deviation
                b = np.random.normal(mu, sigma, pattern_len)
            output = random.randint(0, 1)
            # outputs.append(output)
            patterns.append((b, output))
        return patterns

    def add_input_noise(cls, pattern, input_noise, scaled_noise=False):
        if input_noise > 0:
            pattern = copy.deepcopy(pattern)
            if scaled_noise:
                p0 = 1-input_noise
                for i in range(len(pattern)):
                    r = random.random()
                    pattern[i] = pattern[i]*p0 + r*input_noise
            else:
                for i in range(len(pattern)):
                    if random.random() < input_noise:
                        pattern[i] = random.random()
        return pattern

    def train(
        self,
        patterns,
        n_iteration=None,
        # input_noise=None,
        seed=0
        ):

        if n_iteration is None:
            n_iteration = len(patterns)*10
        # if n_iteration < self.min_train_it:
        #     n_iteration = self.min_train_it
        # if input_noise is None:
        #     input_noise = self.default_input_noise

        # stats
        activated_grcs = 0
        random.seed(seed)

        for i in range(n_iteration):
            # print(patterns[random.randint(0, len(patterns)-1)])
            ind = random.randint(0, len(patterns)-1)
            # print(ind)
            # print(patterns[ind])
            pattern, output = patterns[ind]
            pattern = self.add_input_noise(pattern, input_noise)
            for grc in self.grcs:
                grc.train(pattern, output)
                if grc.activated:
                    activated_grcs += 1

            # if i % 1000 == 0:
            #     print(f'{i}..')

        activated_grcs_level = activated_grcs / len(self.grcs) / n_iteration
        logger.debug(f'activated_grcs_level: {activated_grcs_level} ({activated_grcs / n_iteration} grcs out of {len(self.grcs)})')

    def encode(self, input_pattern, out_array=None):
        if out_array is None:
            out_array = np.empty(len(self.grcs), dtype=np.uint8)
        for i, grc in enumerate(self.grcs):
            if grc.activate(input_pattern):
                out_array[i] = 1
            else:
                out_array[i] = 0
        return out_array

    def evaluate(
        self,
        patterns,
        n_iteration=None,
        no_random=False,
        # input_noise=None,
        # decoder_error_margin=None,
        seed=0,
        calibrate_activation_level=False,
        # output_act_lv=False,
        ):
        if n_iteration is None:
            n_iteration = 10*len(patterns)
        n_iteration = max(self.min_eval_it, n_iteration)

        if no_random:
            n_iteration = len(patterns)
        self.reset()

        # for grc in self.grcs[0:20]:
        #     print(f'len: {len(grc.claws)}, scale: {grc.act_lv_scale:.2f}')

        random.seed(seed)
        for i in range(n_iteration):
            if no_random:
                pattern, output = patterns[i]
            else:
                pattern, output = patterns[random.randint(0, len(patterns)-1)]
            self.set_mfs_pattern(pattern)
            for grc in self.grcs:
                act = grc.activate(pattern)
        if calibrate_activation_level is not False:
            self.calibrate_grc_activation_level(calibrate_activation_level)
        return


    def print_grc_weights(self, count=200):

        weights = []
        for i, grc in enumerate(self.grcs):
            weights.append(grc.output_weight)
            if i > count:
                break
        print(weights)

    def set_mfs_pattern(self, pattern):
        for mf in self.mfs:
            mf.activate(pattern)

    def get_mfs_activities(self):
        # ret = []
        # for mf in self.mfs:
        #     ret.append(mf.activations)

        for mf in self.mfs:
            xlen = len(self.mfs)
            ylen = len(mf.activations)
            break
        ret = np.empty((ylen, xlen), dtype=np.float32)
        for i, mf in enumerate(self.mfs):
            for j, val in enumerate(mf.activations):
                ret[j][i] = val

        return ret

    def get_grc_activities(self):
        # ret = []
        # for grc in self.grcs:
        #     ret.append(grc.activations)
        # return ret

        for mf in self.grcs:
            xlen = len(self.grcs)
            ylen = len(mf.activations)
            break
        ret = np.empty((ylen, xlen), dtype=np.uint8)
        for i, mf in enumerate(self.grcs):
            for j, val in enumerate(mf.activations):
                ret[j][i] = val

        return ret


    def calibrate_grc_activation_level(self, act_lv=None):
        if act_lv is None:
            act_lv = self.act_lv
        for grc in self.grcs:
            grc.calibrate_activation_level(act_lv)

    def add_noise_patterns(
            self, patterns, prob, n, seed=None, scaled_noise=False):
        if seed is not None:
            random.seed(seed)
        out_arr = []
        for pattern_output in patterns:
            # print(pattern_output)
            pattern, output = pattern_output
            for i in range(n):
                new_pattern = self.add_input_noise(pattern, prob, scaled_noise)
                out_arr.append((new_pattern, output))
        return out_arr

    def print_grc_act_lv_scale(self):
        # scales = []
        # for grc in self.grcs:
        #     scales.append(grc.act_lv_scale)
        # print(scales)
        print([grc.act_lv_scale for grc in self.grcs])


from collections import defaultdict
import itertools
def count_redundancy(g):
    pos = 0
    grcs_claws = []
    mf_to_grcs = defaultdict(set)
    for grc_id, dendrite_count in enumerate(g.dendrite_counts):
        claws = []
        for j in range(dendrite_count):
            mf_id = g.dendrite_mf_map[pos]
            pos += 1
            claws.append(mf_id)
            mf_to_grcs[mf_id].add(grc_id)
        grcs_claws.append(set(claws))
    nshares = defaultdict(int)
    for mf_id, grcs in mf_to_grcs.items():
        for pair in itertools.combinations(grcs, 2):
            nshare = len(grcs_claws[pair[0]] & grcs_claws[pair[1]])
            nshares[nshare] += 1
    for n in sorted(nshares.keys()):
        print(f'{n}: {nshares[n]/len(g.dendrite_counts)}')

# count_redundancy(sim_lite)


def generate_binary_patterns(pattern_len, count, f):
    patterns = []
    # np.random.seed(seed)
    # random.seed(seed)
    threshold = int(pattern_len*f+0.5)
    base = np.zeros(pattern_len, dtype=np.uint8)
    base[0:threshold] = 1
    for i in range(count):
        np.random.shuffle(base)
        b = base.copy()
        output = random.randint(0, 1)
        patterns.append((b, output))
    return patterns

# def add_input_noise(pattern, input_noise):
#     if input_noise > 0:
#         pattern = copy.deepcopy(pattern)
#         for i in range(len(pattern)):
#             if random.random() < input_noise:
#                 # pattern[i] = not pattern[i]
#                 pattern[i] = random.random()
#     return pattern

# def add_noise_binary_patterns(pattern, prob, f=None, n=1, seed=0):
#     if f is None:
#         f = pattern.sum() / len(pattern)
#     ones = []
#     zeros = []
#     for i, b in enumerate(pattern):
#         if b:
#             ones.append(i)
#         else:
#             zeros.append(i)
#     ones = np.array(ones, dtype=np.uint32)
#     zeros = np.array(zeros, dtype=np.uint32)
#     ret = []
#     num_flips = int(prob*f*len(pattern)+.5)
#     for i in range(n):
#         new_pat = pattern.copy()
#         np.random.shuffle(ones)
#         for j in range(num_flips):
#             new_pat[ones[j]] = 0
#         np.random.shuffle(zeros)
#         for j in range(num_flips):
#             new_pat[zeros[j]] = 1
#         ret.append(new_pat)
#     return ret

def add_noise_binary_patterns(pattern, prob, f=None, n=1, seed=0):
    if f is None:
        f = pattern.sum() / len(pattern)
    ret = []
    for i in range(n):
        noisy_pattern = copy.deepcopy(pattern)
        for i in range(len(noisy_pattern)):
            if random.random() < prob:
                r = random.random()
                if r < f:
                    noisy_pattern[i] = 1
                else:
                    noisy_pattern[i] = 0
        ret.append(noisy_pattern)
    return ret

def generate_random_pattern(pattern_len, type='random'):
    b = [None]*pattern_len
    for k in range(pattern_len):
        b[k] = random.random()
    return b


def make_noisy_patterns_float(
        patterns, prob, n, seed=None, scaled_noise=False, signal_mask=None):
    if signal_mask:
        assert not scaled_noise
    if seed is not None:
        random.seed(seed)
    out_arr = []
    for pattern_output in patterns:
        # print(pattern_output)
        pattern, output = pattern_output
        for i in range(n):
            new_pattern = add_input_noise_float(pattern, prob, scaled_noise, signal_mask)
            out_arr.append((new_pattern, output))
    return out_arr

def add_input_noise_float(
    pattern, input_noise, scaled_noise=False, signal_mask=None):
    if input_noise > 0:
        pattern = copy.deepcopy(pattern)
        if scaled_noise:
            p0 = 1-input_noise
            for i in range(len(pattern)):
                r = random.random()
                pattern[i] = pattern[i]*p0 + r*input_noise
        elif signal_mask:
            for i in range(len(pattern)):
                if not signal_mask[i]:
                    if random.random() < input_noise:
                        pattern[i] = random.random()
        else:
            for i in range(len(pattern)):
                if random.random() < input_noise:
                    pattern[i] = random.random()
    return pattern

def add_noise_to_core_patterns(
        patterns, prob, n,
        seed=None):
    if seed:
        random.seed(seed)
        np.random.seed(seed)
    out_arr = []
    pattern_len = len(patterns[0][0])
    assert pattern_len <= 65535

    noise_mask_len = int(prob*pattern_len+0.5)
    noise_mask = np.zeros(pattern_len, dtype=np.uint8)
    noise_mask[0:noise_mask_len] = 1

    for pattern_output in patterns:
        pattern, output = pattern_output
        np.random.shuffle(noise_mask)
        for i in range(n):
            random_pat = copy.deepcopy(pattern)
            for j in range(pattern_len):
                if noise_mask[j]:
                    random_pat[j] = random.random()
            out_arr.append((random_pat, output))
    return out_arr

# def add_noise_to_core_patterns(
#         patterns, prob, n,
#         seed=0):
#     random.seed(seed)
#     np.random.seed(seed)
#     out_arr = []
#     pattern_len = len(patterns[0][0])
#     core_len = int(pattern_len*(1-prob))
#     random_core_indices = [k for k in range(pattern_len)]
#     assert pattern_len <= 65535
#     random_core_indices = np.array(random_core_indices, dtype=np.uint16)
#     for pattern_output in patterns:
#         pattern, output = pattern_output
#         np.random.shuffle(random_core_indices)
#         for i in range(n):
#             random_pat = generate_random_pattern(pattern_len)
#             for j in range(core_len):
#                 random_pat[random_core_indices[j]] = pattern[j]
#             out_arr.append((random_pat, output))
#     return out_arr




