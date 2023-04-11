class CustomLSDs(AddLocalShapeDescriptor):
    def __init__(self, segmentation, descriptor, *args, **kwargs):
        super().__init__(segmentation, descriptor, *args, **kwargs)
        self.extractor = LsdExtractor(self.sigma[0:2], self.mode, self.downsample)
    def process(self, batch, request):
        labels = batch[self.segmentation].data
        spec = batch[self.segmentation].spec.copy()
        spec.dtype = np.float32
        descriptor = np.zeros(shape=(6, *labels.shape))
        for z in range(labels.shape[0]):
            labels_sec = labels[z]
            descriptor_sec = self.extractor.get_descriptors(
                segmentation=labels_sec, voxel_size=spec.voxel_size[1:]
            )
            descriptor[:, z] = descriptor_sec
        batch = Batch()
        batch[self.descriptor] = Array(descriptor.astype(spec.dtype), spec)
        return batch