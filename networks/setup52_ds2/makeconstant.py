# import logging
# import random

# import numpy as np

from gunpowder import BatchFilter

# logger = logging.getLogger(__name__)


class MakeConstant(BatchFilter):

    def __init__(
            self,
            key,
            value
            ):

        self.key = key
        self.value = value

    def process(self, batch, request):

        batch.arrays[self.key].data[:] = self.value
