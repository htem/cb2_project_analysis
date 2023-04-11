# import logging
# import random

# import numpy as np

from gunpowder import BatchFilter

# logger = logging.getLogger(__name__)


class SetConstantValue(BatchFilter):

    def __init__(
            self,
            key,
            ):

        self.key = key
        self.mask = logical_and_mask
        self.mask_val = logical_and_mask_val

    # def setup(self):
    #     if self.mask:
    #         assert self.mask in self.spec, (
    #             "Reject can only be used if %s is provided" % self.mask)
    #     if self.ensure_nonempty:
    #         assert self.ensure_nonempty in self.spec, (
    #             "Reject can only be used if %s is provided" %
    #             self.ensure_nonempty)
    #     self.upstream_provider = self.get_upstream_provider()

    def process(self, batch, request):

        array = batch.arrays[self.key].data
        mask_array = batch.arrays[self.mask].data

        array[mask_array != self.mask_val] = 0
        batch.arrays[self.key].data = array
