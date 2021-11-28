import random
import math
import copy
import numpy as np
import logging
import collections


class SimulationLite():

    def __init__(self, sim):
        self.dendrite_counts = []
        self.dendrite_mf_map = []
        self.thresholds = []
        for grc in sim.grcs:
            self.dendrite_counts.append(len(grc.claws))
            self.thresholds.append(grc.act_lv_scale)
            for claw in grc.claws:
                assert claw <= 65535
                self.dendrite_mf_map.append(claw)

        # self.dendrite_mf_map = np.array(self.dendrite_mf_map, dtype=np.uint16)

    def encode(self, input_pattern, out_array=None):
        if out_array is None:
            out_array = np.empty(len(self.dendrite_counts), dtype=np.uint8)

        dendrite_pos = 0
        for i, dendrite_count in enumerate(self.dendrite_counts):
            s = 0.0
            for j in range(dendrite_count):
                s += input_pattern[self.dendrite_mf_map[dendrite_pos]]
                dendrite_pos += 1
            if s >= self.thresholds[i]:
                out_array[i] = 1
            else:
                out_array[i] = 0
        return out_array

