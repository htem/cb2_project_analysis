import logging
import random
import math
# import numpy as np

from gunpowder import BatchFilter
# from gunpowder.coordinate import Coordinate

logger = logging.getLogger(__name__)

class DuplicateAugment(BatchFilter):
    '''Randomly mirror and transpose all :class:`Arrays<Array>` and
    :class:`Points` in a batch.

    Args:
    '''

    def __init__(self,
            label_key,
            # raw_key,
            voxel_size,
            max_consecutive_duplicate=6,
            prob_duplicate=0,
            prob_edge_duplicate=0,
            ):

        self.label_key = label_key
        # self.raw_key = raw_key
        self.max_consecutive_duplicate = max_consecutive_duplicate
        self.prob_duplicate = prob_duplicate
        self.prob_edge_duplicate = prob_edge_duplicate
        self.skip_duplicate = False
        self.voxel_size = voxel_size

    def setup(self):

        self.dims = self.spec.get_total_roi().dims()

        # mirror_mask and transpose_dims refer to the indices of the spatial
        # dimensions only, starting counting at 0 for the first spatial
        # dimension


    def prepare(self, request):

        spec = request[self.label_key]
        # print("request:", request)
        self.label_roi = spec.roi

        self.edge_duplicate = random.random() < self.prob_edge_duplicate

        # raw_voxel_size = spec.voxel_size

        # if raw_voxel_size is None:
        #     self.skip_duplicate = True
        # else:
        #     self.skip_duplicate = False

        # if not self.skip_duplicate and self.prob_duplicate > 0:
        if self.prob_duplicate > 0:
            roi = self.label_roi
            # print("roi.get_begin():", roi.get_begin())
            # print("raw_voxel_size:", self.voxel_size)

            begin = int(roi.get_begin()[0] / self.voxel_size[0])
            end = int(roi.get_end()[0] / self.voxel_size[0])

            self.duplicate_map = [i for i in range(end-begin-1)]
            self.duplicate_map_offset = begin

            duplicate_count = 0
            duplicate_num = 0
            c_duplicate_begin = None
            c_duplicate_end = None

            for c in range(end-begin-1):

                if duplicate_count:
                    if duplicate_count > int(duplicate_num/2):
                        self.duplicate_map[c] = c_duplicate_begin
                    else:
                        self.duplicate_map[c] = c_duplicate_end
                    duplicate_count -= 1
                    continue

                r = random.random()
                if r < self.prob_duplicate:
                    c_duplicate_begin = c
                    duplicate_num = math.ceil(random.random()*self.max_consecutive_duplicate)
                    c_duplicate_end = duplicate_num + c + 1
                    if not c_duplicate_end > len(self.duplicate_map)-1:
                        duplicate_count = duplicate_num
                    else:
                        duplicate_count = 0

                self.duplicate_map[c] = c

            # print(self.duplicate_map)


        # t = list(self.transpose_dims)
        # random.shuffle(t)
        # self.transpose = list(range(self.dims))
        # for o, n in zip(self.transpose_dims, t):
        #     self.transpose[o] = n

        # logger.debug("mirror = " + str(self.mirror))
        # logger.debug("transpose = " + str(self.transpose))

        # reverse_transpose = [0]*self.dims
        # for d in range(self.dims):
        #     reverse_transpose[self.transpose[d]] = d

        # logger.debug("downstream request = " + str(request))

        # self.__transpose_request(request, reverse_transpose)
        # self.__mirror_request(request, self.mirror)

        # logger.debug("upstream request = " + str(request))

    def process(self, batch, request):

        # edge_duplicate
        if self.edge_duplicate:
            for (array_key, array) in batch.arrays.items():
                if array_key not in request:
                    continue
                voxel_size = array.spec.voxel_size
                low_start = int(self.label_roi.get_begin()[0]/voxel_size[0])+1
                low_end = int(array.spec.roi.get_begin()[0]/voxel_size[0])
                hi_start = int(self.label_roi.get_end()[0]/voxel_size[0])-1
                hi_end = int(array.spec.roi.get_end()[0]/voxel_size[0])
                # print("low_start:", low_start)
                # print("low_end:", low_end)
                # print("hi_start:", hi_start)
                # print("hi_end:", hi_end)

                for c in range(low_start-1, low_end-1, -1):
                    array.data[c] = array.data[low_start]
                for c in range(hi_start+1, hi_end, 1):
                    array.data[c] = array.data[hi_start]

        # duplicate sections
        # if not self.skip_duplicate and self.prob_duplicate > 0:
        if self.prob_duplicate > 0:
            for (array_key, array) in batch.arrays.items():
                if array_key not in request:
                    continue
                # voxel_size = array.spec.voxel_size
                offset = self.duplicate_map_offset - int(array.spec.roi.get_begin()[0]/self.voxel_size[0])
                offset += 1

                # print("array_key:", array_key)
                # print("offset:", offset)
                for c in range(len(self.duplicate_map)):
                    # print("%d -> %d" % (c+offset, self.duplicate_map[c]+offset))
                    array.data[c+offset] = array.data[self.duplicate_map[c]+offset]
        # else:
        #     assert False
