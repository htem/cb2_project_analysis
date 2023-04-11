import logging
import numpy as np

from gunpowder import BatchFilter
from gunpowder.coordinate import Coordinate
from gunpowder.array import Array

logger = logging.getLogger(__name__)


def seg_to_affgraph(seg, nhood):

    nhood = np.array(nhood)

    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = seg.shape
    nEdge = nhood.shape[0]
    dims = nhood.shape[1]
    aff = np.zeros((nEdge,) + shape, dtype=np.int32)

    if dims == 2:

        for e in range(nEdge):
            aff[e, \
                max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] = \
                            (seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] == \
                             seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                                max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])] ) \
                            * ( seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] > 0 ) \
                            * ( seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                                max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])] > 0 )

    elif dims == 3:

        for e in range(nEdge):
            aff[e, \
                max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] = \
                            (seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                                max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] == \
                             seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                                max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                                max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] ) \
                            * ( seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                                max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] > 0 ) \
                            * ( seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                                max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                                max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] > 0 )

    else:

        raise RuntimeError(
            f"AddAffinities works only in 2 or 3 dimensions, not {dims}")

    return aff

class AddAffinities(BatchFilter):
    '''Add an array with affinities for a given label array and neighborhood to 
    the batch. Affinity values are created one for each voxel and entry in the 
    neighborhood list, i.e., for each voxel and each neighbor of this voxel. 
    Values are 1 iff both labels (of the voxel and the neighbor) are equal and 
    non-zero.

    Args:

        affinity_neighborhood (``list`` of array-like):

            List of offsets for the affinities to consider for each voxel.

        labels (:class:`ArrayKey`):

            The array to read the labels from.

        affinities (:class:`ArrayKey`):

            The array to generate containing the affinities.

        labels_mask (:class:`ArrayKey`, optional):

            The array to use as a mask for ``labels``. Affinities connecting at
            least one masked out label will be masked out in
            ``affinities_mask``. If not given, ``affinities_mask`` will contain
            ones everywhere (if requested).

        unlabeled (:class:`ArrayKey`, optional):

            A binary array to indicate unlabeled areas with 0. Affinities from
            labelled to unlabeled voxels are set to 0, affinities between
            unlabeled voxels are masked out (they will not be used for
            training).

        affinities_mask (:class:`ArrayKey`, optional):

            The array to generate containing the affinitiy mask, as derived
            from parameter ``labels_mask``.
    '''

    def __init__(
            self,
            affinity_neighborhood,
            labels,
            affinities,
            labels_mask=None,
            unlabeled=None,
            unlabeled_z_fix=False,
            affinities_mask=None):

        self.affinity_neighborhood = np.array(affinity_neighborhood)
        self.labels = labels
        self.unlabeled = unlabeled
        self.labels_mask = labels_mask
        self.affinities = affinities
        self.affinities_mask = affinities_mask
        self.unlabeled_z_fix = unlabeled_z_fix

    def setup(self):

        assert self.labels in self.spec, (
            "Upstream does not provide %s needed by "
            "AddAffinities"%self.labels)

        voxel_size = self.spec[self.labels].voxel_size

        dims = self.affinity_neighborhood.shape[1]
        self.padding_neg = Coordinate(
                min([0] + [a[d] for a in self.affinity_neighborhood])
                for d in range(dims)
        )*voxel_size

        self.padding_pos = Coordinate(
                max([0] + [a[d] for a in self.affinity_neighborhood])
                for d in range(dims)
        )*voxel_size

        logger.debug("padding neg: " + str(self.padding_neg))
        logger.debug("padding pos: " + str(self.padding_pos))

        spec = self.spec[self.labels].copy()
        if spec.roi is not None:
            spec.roi = spec.roi.grow(self.padding_neg, -self.padding_pos)
        # spec.dtype = np.float32
        spec.dtype = np.uint8

        self.provides(self.affinities, spec)
        if self.affinities_mask:
            self.provides(self.affinities_mask, spec)
        self.enable_autoskip()

    def prepare(self, request):

        if self.labels_mask:
            assert (
                request[self.labels].roi ==
                request[self.labels_mask].roi),(
                "requested GT label roi %s and GT label mask roi %s are not "
                "the same."%(
                    request[self.labels].roi,
                    request[self.labels_mask].roi))

        if self.unlabeled:
            assert (
                request[self.labels].roi ==
                request[self.unlabeled].roi),(
                "requested GT label roi %s and GT unlabeled mask roi %s are not "
                "the same."%(
                    request[self.labels].roi,
                    request[self.unlabeled].roi))

        if self.labels not in request:
            request[self.labels] = request[self.affinities].copy()

        labels_roi = request[self.labels].roi
        context_roi = request[self.affinities].roi.grow(
            -self.padding_neg,
            self.padding_pos)

        # grow labels ROI to accomodate padding
        labels_roi = labels_roi.union(context_roi)
        request[self.labels].roi = labels_roi

        # same for label mask
        if self.labels_mask and self.labels_mask in request:
            request[self.labels_mask].roi = \
                request[self.labels_mask].roi.union(context_roi)

        # and unlabeled mask
        if self.unlabeled and self.unlabeled in request:
            request[self.unlabeled].roi = \
                request[self.unlabeled].roi.union(context_roi)

        logger.debug("upstream %s request: "%self.labels + str(labels_roi))

    def process(self, batch, request):

        affinities_roi = request[self.affinities].roi

        logger.debug("computing ground-truth affinities from labels")
        affinities = seg_to_affgraph(
                batch.arrays[self.labels].data.astype(np.int32),
                self.affinity_neighborhood
        ).astype(np.uint8)

        # crop affinities to requested ROI
        offset = affinities_roi.get_offset()
        shift = -offset - self.padding_neg
        crop_roi = affinities_roi.shift(shift)
        crop_roi /= self.spec[self.labels].voxel_size
        crop = crop_roi.get_bounding_box()

        logger.debug("cropping with " + str(crop))
        affinities = affinities[(slice(None),)+crop]

        spec = self.spec[self.affinities].copy()
        spec.roi = affinities_roi
        batch.arrays[self.affinities] = Array(affinities, spec)

        if self.affinities_mask and self.affinities_mask in request:

            if self.labels_mask:

                logger.debug("computing ground-truth affinities mask from "
                             "labels mask")
                affinities_mask = seg_to_affgraph(
                    batch.arrays[self.labels_mask].data.astype(np.int32),
                    self.affinity_neighborhood)
                affinities_mask = affinities_mask[(slice(None),)+crop]

            else:

                affinities_mask = np.ones_like(affinities)

            if self.unlabeled:

                if self.unlabeled_z_fix:
                    # z-affinity is masked out between labeleld and unlabeled voxels
                    unlabeled = np.copy(batch.arrays[self.unlabeled].data)
                    unlabeled_mask = seg_to_affgraph(
                        unlabeled.astype(np.int32),
                        self.affinity_neighborhood)
                    unlabeled_mask = unlabeled_mask[(slice(None),)+crop]

                    # combine with mask, but only for z-dim
                    # assuming z first. TODO: use neighborhood
                    affinities_mask[0] = affinities_mask[0]*unlabeled_mask[0]

                # 1 for all affinities between unlabeled voxels
                unlabeled = (1 - batch.arrays[self.unlabeled].data)
                unlabeled_mask = seg_to_affgraph(
                    unlabeled.astype(np.int32),
                    self.affinity_neighborhood)
                unlabeled_mask = unlabeled_mask[(slice(None),)+crop]

                # 0 for all affinities between unlabeled voxels
                unlabeled_mask = (1 - unlabeled_mask)

                # combine with mask
                affinities_mask = affinities_mask*unlabeled_mask

            # affinities_mask = affinities_mask.astype(np.float32)
            affinities_mask = affinities_mask.astype(np.uint8)
            batch.arrays[self.affinities_mask] = Array(affinities_mask, spec)

        else:

            if self.labels_mask is not None:
                logger.warning("GT labels does have a mask, but affinities "
                               "mask is not requested.")

        # crop labels to original label ROI
        if self.labels in request:
            roi = request[self.labels].roi
            batch.arrays[self.labels] = batch.arrays[self.labels].crop(roi)

        # same for label mask
        if self.labels_mask and self.labels_mask in request:
            roi = request[self.labels_mask].roi
            batch.arrays[self.labels_mask] = \
                batch.arrays[self.labels_mask].crop(roi)

        # and unlabeled mask
        if self.unlabeled and self.unlabeled in request:
            roi = request[self.unlabeled].roi
            batch.arrays[self.unlabeled] = \
                batch.arrays[self.unlabeled].crop(roi)

        batch.affinity_neighborhood = self.affinity_neighborhood
