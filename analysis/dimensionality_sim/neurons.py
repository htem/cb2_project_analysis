import random
import math
import copy
import numpy as np
import logging
import collections
from collections import defaultdict
import itertools


logger = logging.getLogger(__name__)


class GranuleCell():
    def __init__(
        self,
        claws,
        claw_weights=None,
    ):
        self.activations = []
        self.inputs = []
        self.claws = claws
        self.claws.sort()
        if claw_weights:
            self.claw_weights = claw_weights
            assert False, "Untested"
        else:
            self.claw_weights = [1]*len(self.claws)
        self.activated = False
        self.act_lv_scale = 1
        self.broken = False
        self.binary_mode = False
        self.calibrated = False

    def activate(
            self, pattern,
            ):
        if self.broken:
            self.activations.append(0)
            self.activated = False
            return False
        sum = 0
        for i, claw in enumerate(self.claws):
            sum += pattern[claw]
        if not self.calibrated:
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
            assert False
        # print(self.inputs); print(idx); asdf
        if self.binary_mode:
            floor_val = self.inputs[idx]
            floor_indices = np.where(np.array(self.inputs) == floor_val)[0]
            min_idx = min(floor_indices)
            max_idx = max(floor_indices)
            # print(floor_indices)
            interpolated_val = (idx-min_idx) / (max_idx-min_idx) + floor_val
            self.act_lv_scale = interpolated_val
            # if len(self.claws) == 5:
            #     print(self.inputs)
            #     print(self.act_lv_scale)
            #     asdf
        else:
            self.act_lv_scale = self.inputs[idx]
    def set_act_lv_scale(self, calibrated_scales):
        self.act_lv_scale = calibrated_scales[len(self.claws)]
        assert self.act_lv_scale > 0
        # self.act_lv_scale += 1
        if self.binary_mode:
            prob = self.act_lv_scale - int(self.act_lv_scale)
            self.act_lv_scale = int(self.act_lv_scale)
            if random.random() < prob:
                self.act_lv_scale += 1


class MossyFiber():
    def __init__(self, mf_id):
        self.mf_id = mf_id
        self.activations = []
        self.locs = None
        pass
    def reset(self):
        self.activations = []
    def activate(self, pattern):
        self.activations.append(pattern[self.mf_id])


class Simulation():

    def __init__(
        self,
        input_graph,
        min_eval_it=5000,
        per_bouton=False,
    ):

        if per_bouton:
            self.num_mfs = len(input_graph.mf_locs)
        else:
            self.num_mfs = len(input_graph.mfs)
        self.num_grcs = len(input_graph.grcs)
        self.min_eval_it = min_eval_it
        self.mf_size = defaultdict(int)
        self.init_mfs()
        if per_bouton:
            self.init_grcs_per_bouton(input_graph)
        else:
            self.init_grcs(input_graph)
        self.failure_rate = None

    def reset(self):
        for grc in self.grcs:
            grc.reset()
        for mf in self.mfs:
            mf.reset()

    def init_mfs(self):
        self.mfs = []
        for i in range(self.num_mfs):
            self.mfs.append(MossyFiber(mf_id=i))
            self.mf_size[i] = 0

    def init_grcs(self, input_graph):
        self.grcs = []
        mapping = {}
        counter = 0
        self.mf_size = defaultdict(int)
        for mf_id, mf in input_graph.mfs.items():
            mapping[mf_id] = counter
            self.mfs[counter].locs = mf.locs
            counter += 1
        for grc_id, grc in input_graph.grcs.items():
            claws = [mapping[mf_id] for mf_id, _ in grc.edges]
            self.grcs.append(
                GranuleCell(
                    claws=claws,
                    )
                )
            for mapped_mf_id in set(claws):
                self.mf_size[mapped_mf_id] += 1

    def init_grcs_per_bouton(self, input_graph):
        self.grcs = []
        mapping = {}
        counter = 0
        for mf_id, mf in input_graph.mfs.items():
            for loc in mf.locs:
                mapping[loc] = counter
                self.mfs[counter].locs = [loc]
                counter += 1
        for grc_id, grc in input_graph.grcs.items():
            claws = [mapping[loc] for mf_id, loc in grc.edges]
            self.grcs.append(
                GranuleCell(
                    claws=claws,
                    )
                )
            for mapped_mf_id in set(claws):
                self.mf_size[mapped_mf_id] += 1

    # def init_mfs(self):
    #     self.mfs = []
    #     for i in range(self.num_mfs):
    #         self.mfs.append(MossyFiber(mf_id=i))

    # def init_grcs(self, input_graph):
    #     self.grcs = []
    #     mapping = {}
    #     counter = 0
    #     self.mf_size = defaultdict(int)
    #     for mf_id, mf in input_graph.mfs.items():
    #         mapping[mf_id] = counter
    #         self.mfs[counter].locs = mf.locs
    #         counter += 1
    #     for grc_id, grc in input_graph.grcs.items():
    #         claws = [mapping[mf_id] for mf_id, _ in grc.edges]
    #         self.grcs.append(
    #             GranuleCell(
    #                 claws=claws,
    #                 )
    #             )
    #         for mapped_mf_id in set(claws):
    #             self.mf_size[mapped_mf_id] += 1

    def set_failure_rate(self, failure_rate, seed):
        random.seed(seed)
        for grc in self.grcs:
            grc.broken = True if random.random() < failure_rate else False

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
        seed=0
        ):

        if n_iteration is None:
            n_iteration = len(patterns)*10

        # stats
        activated_grcs = 0
        random.seed(seed)

        for i in range(n_iteration):
            ind = random.randint(0, len(patterns)-1)
            pattern, output = patterns[ind]
            pattern = self.add_input_noise(pattern, input_noise)
            for grc in self.grcs:
                grc.train(pattern, output)
                if grc.activated:
                    activated_grcs += 1

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
        seed=0,
        calibrate_activation_level=False,
        ):
        if n_iteration is None:
            n_iteration = 10*len(patterns)
        n_iteration = max(self.min_eval_it, n_iteration)

        if no_random:
            n_iteration = len(patterns)
        self.reset()
        random.seed(seed)
        for i in range(n_iteration):
            if no_random:
                pattern, output = patterns[i]
            else:
                pattern, output = patterns[random.randint(0, len(patterns)-1)]
            if isinstance(pattern, np.ndarray):
                pattern = list(pattern)
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
        for grc in self.grcs:
            xlen = len(self.grcs)
            ylen = len(grc.activations)
            break
        ret = np.empty((ylen, xlen), dtype=np.uint8)
        for i, grc in enumerate(self.grcs):
            for j, val in enumerate(grc.activations):
                ret[j][i] = val

        return ret

    def generate_patterns(
            self,
            count,
            type='uniform',
        ):
        patterns = []
        pattern_len = self.num_mfs

        for i in range(count):
            if type == 'uniform':
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
            pattern, output = pattern_output
            for i in range(n):
                new_pattern = self.add_input_noise(pattern, prob, scaled_noise)
                out_arr.append((new_pattern, output))
        return out_arr

    def print_grc_act_lv_scale(self):
        scales = defaultdict(list)
        for grc in self.grcs:
            scales[len(grc.claws)].append(grc.act_lv_scale)
            # print([grc.act_lv_scale for grc in self.grcs])
        print(scales)

    def get_activation_levels(self):
        scales = defaultdict(list)
        avg_scales = {}
        for grc in self.grcs:
            scales[len(grc.claws)].append(grc.act_lv_scale)
        for nclaws in scales:
            avg_scales[nclaws] = sum(scales[nclaws])/len(scales[nclaws])
        return avg_scales

    def set_activation_levels(self, calibrated_scales):
        for grc in self.grcs:
            grc.set_act_lv_scale(calibrated_scales)
    def set_binary_mode(self):
        for grc in self.grcs:
            grc.binary_mode = True

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

def generate_random_pattern(pattern_len, type='random'):
    b = [None]*pattern_len
    for k in range(pattern_len):
        b[k] = random.random()
    return b


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




