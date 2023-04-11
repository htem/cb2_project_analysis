
import numpy as np
from gunpowder import BatchFilter, Array, BatchRequest, Batch

from add_local_shape_descriptor import AddLocalShapeDescriptor
from local_shape_descriptor import LsdExtractor


class Add2DLocalShapeDescriptor(AddLocalShapeDescriptor):

    def __init__(self, segmentation, descriptor, *args, **kwargs):
        super().__init__(segmentation, descriptor, *args, **kwargs)
        self.extractor = LsdExtractor(self.sigma[0:2], self.mode, self.downsample)

    def process(self, batch, request):
        labels = batch[self.segmentation].data
        spec = batch[self.segmentation].spec.copy()
        spec.dtype = np.float32

        dims = len(self.voxel_size)
        segmentation_array = batch[self.segmentation]
        # get voxel roi of requested descriptors
        # this is the only region in
        # which we have to compute the descriptors
        seg_roi = segmentation_array.spec.roi
        descriptor_roi = request[self.descriptor].roi
        voxel_roi_in_seg = (
            seg_roi.intersect(descriptor_roi) -
            seg_roi.get_offset())/self.voxel_size
        crop = voxel_roi_in_seg.get_bounding_box()

        descriptor = np.zeros(shape=(6, *labels.shape))
        for z in range(labels.shape[0]):
            labels_sec = labels[z]
            descriptor_sec = self.extractor.get_descriptors(
                segmentation=labels_sec, voxel_size=spec.voxel_size[1:]
            )
            descriptor[:, z] = descriptor_sec

        # batch = Batch()
        # batch[self.descriptor] = Array(descriptor.astype(spec.dtype), spec)


        # create descriptor array
        descriptor_spec = self.spec[self.descriptor].copy()
        descriptor_spec.roi = request[self.descriptor].roi.copy()
        # descriptor_array = Array(descriptor, descriptor_spec)
        descriptor_array = Array(descriptor.astype(spec.dtype), spec)

        old_batch = batch

        # Create new batch for descriptor:
        batch = Batch()

        # create mask array
        if self.mask and self.mask in request:


            if self.labels_mask:

                mask = self._create_mask(
                        old_batch,
                        self.labels_mask,
                        descriptor,
                        crop)

            else:

                mask = (segmentation_array.crop(descriptor_roi).data!=0).astype(np.float32)

                mask_shape = len(mask.shape)

                assert mask.shape[-mask_shape:] == \
                        descriptor.shape[-mask_shape:]


                mask = np.array([mask]*descriptor.shape[0])


            if self.unlabeled_mask:

                unlabelled_mask = self._create_mask(
                        old_batch,
                        self.unlabeled_mask,
                        descriptor,
                        crop)

                mask = mask*unlabelled_mask

            batch[self.mask] = Array(
                    mask.astype(spec.dtype),
                    descriptor_spec.copy())

        batch[self.descriptor] = descriptor_array

        return batch
