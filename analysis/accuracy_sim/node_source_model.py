from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.profiling import Timing
from gunpowder.roi import Roi
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec
from gunpowder import BatchProvider
import logging

logger = logging.getLogger(__name__)


class SourceModel(BatchProvider):
    '''
    '''
    def __init__(
            self,
            model,
            datasets,
            batch_size=1,
            ):

        self.model = model
        self.datasets = datasets
        self.batch_size = batch_size
        self.counter = None

    def setup(self):

        spec = ArraySpec()
        spec.voxel_size = Coordinate(1, 1)
        spec.dtype = self.model.get_dtype()
        spec.interpolatable = False

        for (array_key, ds_name) in self.datasets.items():
            self.provides(array_key, spec)

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = Batch()

        if self.counter is None:
            self.counter = request.random_seed
        # outputs = self.model.get(self.batch_size, seed=request.random_seed)
        outputs = self.model.get(self.batch_size, seed=self.counter)
        self.counter += 1

        for (array_key, request_spec) in request.array_specs.items():

            # create array spec
            array_spec = self.spec[array_key].copy()
            array_spec.roi = request_spec.roi

            # add array to batch
            batch.arrays[array_key] = Array(
                outputs[self.datasets[array_key]],
                array_spec)

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch
