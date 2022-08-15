import random
import copy

class Shuffler():
    def __init__(self, dist):
        self.dist = copy.copy(dist)
        random.shuffle(self.dist)
        self.i = 0
    def get_one(self):
        ret = self.dist[self.i]
        self.i += 1
        if self.i == len(self.dist):
            self.i = 0
            random.shuffle(self.dist)
        return ret

class GranuleCell():
    def __init__(self):
        self.edges = []

class MossyFiber():
    def __init__(self, id):
        self.locs = [id]

class GlobalRandomModel():
    def __init__(
            self,
            n_grcs,
            n_mfs,
            n_dendrites=None,
            mf_size_dist=None,
            seed=None,
            ):
        self.n_grcs = n_grcs
        self.n_mfs = n_mfs
        if n_dendrites is None:
            n_dendrites = [4, 4, 5, 5]
        if seed is not None:
            random.seed(seed)
        self.n_dendrites = Shuffler(n_dendrites)
        self.mf_locs = [None]*self.n_mfs

        if mf_size_dist is not None:
            mf_size_shuffler = Shuffler(mf_size_dist)
            self.mf_ids = []
            for i in range(self.n_mfs):
                self.mf_ids.extend([i]*mf_size_shuffler.get_one())
        else:
            self.mf_ids = list(range(self.n_mfs))

        self.randomize(seed)

    def randomize(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.grcs = {}
        used_mfs = set()
        for grc_id in range(self.n_grcs):
            grc = GranuleCell()
            for i in range(self.n_dendrites.get_one()):
                mf_id = random.choice(self.mf_ids)
                grc.edges.append((mf_id, mf_id))
                used_mfs.add(mf_id)
            self.grcs[grc_id] = grc

        print(f'n_grcs: {len(self.grcs)}')
        self.mfs = {}
        for mf_id in used_mfs:
            self.mfs[mf_id] = MossyFiber(mf_id)
        print(f'n_mfs: {len(self.mfs)}')

    def remove_empty_mfs(self):
        return []
