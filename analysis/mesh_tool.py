import os
import struct
# import numpy as np
import daisy
import sys

def getHierarchicalMeshPath(object_id, hierarchical_size):#finds the path to mesh files based on segment numbers
    assert object_id != 0
    level_dirs = []
    num_level = 0
    while object_id > 0:
        level_dirs.append(int(object_id % hierarchical_size))
        object_id = int(object_id / hierarchical_size)
    num_level = len(level_dirs) - 1
    level_dirs = [str(lv) for lv in reversed(level_dirs)]
    return os.path.join(str(num_level), *level_dirs)

base = '/n/balin_tank_ssd1/htem/Segmentation/cb2_v4/output.zarr/meshes/precomputed/mesh/'
if not os.path.exists(base):
    raise RuntimeError("Mesh directory does not exist!!!")

def getMeshVertices(segmentNum, downsample=None, missing_ok=False):
    #opens mesh file from local directory and parses it, returning a trimesh object
    #define server path *EDIT THIS*
    workfile = base + getHierarchicalMeshPath(int(segmentNum), 10000)
    # totalSize = os.stat(workfile).st_size
    # vertices = []
    vertices = set()
    try:
        with open(workfile, 'rb') as f:
            num_vertices = struct.unpack('<I', memoryview(f.read(4)))[-1]
            # vertices = np.empty((num_vertices,3))
            for i in range(num_vertices):
                coord = struct.unpack('<fff', memoryview(f.read(12)))
                if downsample:
                    coord = [int(i/j) for i, j in zip(coord, downsample)]
                    coord = [i*j for i, j in zip(coord, downsample)]
                vertices.add(tuple([int(k) for k in coord]))
    except IOError as e:
        if not missing_ok:
            raise e
        # num_triangles = int((totalSize - (num_vertices*12 + 4))/12)
        # triangles = np.empty((num_triangles,3))
        # for i in range(num_triangles):
        #     triangles[i,] = struct.unpack('<III', memoryview(f.read(12)))
    return vertices

meshHierarchical_size=10000
voxel_size=(40, 8, 8)
find_segment_block_size=(4000, 4096, 4096)
super_block_size=(4000, 8192, 8192)
fragments_block_size=(400, 2048, 2048)
super_offset_hack=(2800, 0, 0)
daisy_block_id_add_one_fix=True

def check_blocksize_consistency(big_bs, small_bs):
    assert len(big_bs) == len(small_bs)
    for a, b in zip(big_bs, small_bs):
        assert a % b == 0

# init variables for box_id calculations
check_blocksize_consistency(find_segment_block_size, fragments_block_size)
check_blocksize_consistency(super_block_size, fragments_block_size)
fragments_block_size = daisy.Coordinate(fragments_block_size)
find_segment_block_size = daisy.Coordinate(find_segment_block_size)
super_block_size = daisy.Coordinate(super_block_size)
super_offset_hack = daisy.Coordinate(super_offset_hack)
check_blocksize_consistency(super_offset_hack, fragments_block_size)
super_offset_frag_nblock = super_offset_hack // fragments_block_size
size_of_voxel = daisy.Roi((0,)*3, voxel_size).size()
fragments_block_roi = daisy.Roi((0,)*3, fragments_block_size)
num_voxels_in_fragment_block = fragments_block_roi.size()//size_of_voxel
super_offset_frag_nblock = super_offset_frag_nblock
local_chunk_size = find_segment_block_size // fragments_block_size
super_chunk_size = super_block_size // find_segment_block_size
fragments_block_size = fragments_block_size
voxel_size = voxel_size
find_segment_block_size = find_segment_block_size
super_block_size = super_block_size

def getBoxId(fragment_id):
    super_id = int(fragment_id)
    # print("super_id:", super_id)
    block_id = int(super_id / num_voxels_in_fragment_block)
    # print("block_id:", block_id)
    fragment_index = daisy.Coordinate(daisy.Block.id2index(block_id))
    # print("fragment_index:", fragment_index)
    fragment_index -= super_offset_frag_nblock
    # print("adjusted fragment_index:", fragment_index)
    local_index = fragment_index // local_chunk_size
    # return local_index
    # print("local_chunk_size:", local_chunk_size)
    # print("local_index:", local_index)
    super_index = local_index // super_chunk_size
    # print("super_chunk_size:", super_chunk_size)
    # print("super_index:", super_index)
    return super_index

def computeVertexDist(u, v):
    return (
        (
            abs(u[0] - v[0]) +
            abs(u[1] - v[1]) +
            abs(u[2] - v[2])
        ),
        (
            abs(u[0] - v[0]),
            abs(u[1] - v[1]),
            abs(u[2] - v[2])
        )
    )



def withinThreshold(u, v, thresholds):
    for i, j, k in zip(u, v, thresholds):
        if abs(i-j) > k:
            return False
    return True

def getClosestVertex(mesh_ids, boxed_vertices, thresholds=(300, 300, 300)):

    min_dist = 100000000
    min_dist_xyz = None
    min_id = None
    for mesh_id in mesh_ids:
        boxid = getBoxId(mesh_id)
        # print(boxid)
        vertices = boxed_vertices[boxid]
        # print(vertices)
        if len(vertices) == 0:
            continue
        try:
            mesh_vertices = getMeshVertices(mesh_id)
        except IOError:
            continue
        for u in mesh_vertices:
            for v in vertices:
                if withinThreshold(u, v, thresholds):
                    dist, dist_xyz = computeVertexDist(u, v)
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_xyz = dist_xyz
                        min_id = v

    return min_id, min_dist, min_dist_xyz


def downsampleVertices(
        vertices, ds, mesh_voxel_size,
        pc_vertices_ds_reverse_cache
        ):
    out = set()
    if ds == (1, 1, 1):
        return set(vertices)
    else:
        for v in vertices:
            v_ds = tuple([int(k/f/v) for k, f, v in zip(v, ds, mesh_voxel_size)])
            out.add(v_ds)
            pc_vertices_ds_reverse_cache[v_ds].add(v)
    return out

def downsampleVertex(v, ds, mesh_voxel_size):
    return tuple([int(k/f/v) for k, f, v in zip(v, ds, mesh_voxel_size)])


def getClosestVertexPyramid(
        mesh_ids, pc_vertices,
        pc_vertices_ds_cache,
        pc_vertices_ds_reverse_cache,
        ds_factors,
        mesh_voxel_size,
        cutoff=0):
    min_ds = (sys.maxsize, sys.maxsize, sys.maxsize)
    if len(pc_vertices) == 0:
        return None, sys.maxsize 
    min_dist = sys.maxsize
    min_pc_vert = None
    # closest_input_coord = None
    # processed = set()
    for mesh_id in mesh_ids:
        try:
            mesh_vertices = getMeshVertices(mesh_id)
        except IOError as e:
            # print(f"IOError: {e}")
            continue
        for input_coord in mesh_vertices:
            for ds in ds_factors:
                if ds[0] > min_ds[0]:
                    continue
                if ds not in pc_vertices_ds_cache:
                    pc_vertices_ds_cache[ds] = downsampleVertices(pc_vertices, ds, mesh_voxel_size, pc_vertices_ds_reverse_cache)
                input_coord_ds = downsampleVertex(input_coord, ds, mesh_voxel_size)
                # print(input_coord)
                if input_coord_ds in pc_vertices_ds_cache[ds]:
                    closest_pc_vertices = pc_vertices_ds_reverse_cache[input_coord_ds]
                    for pc_vert in closest_pc_vertices:
                        dist, dist_xyz = computeVertexDist(pc_vert, input_coord)
                        if min_dist > dist:
                            min_dist = dist
                            min_pc_vert = pc_vert
                        if dist <= cutoff:
                            return min_pc_vert, min_dist
                    # min_ds = ds
                    # closest_input_coord = input_coord
    # if closest_input_coord:
    #     # closest_input_coord = closest_input_coord
    #     ds_val = downsampleVertex(closest_input_coord, min_ds, mesh_voxel_size)
    #     for pc_coord in pc_vertices:
    #         if downsampleVertex(pc_coord, min_ds, mesh_voxel_size) == ds_val:
    #             closest_pc_coord = pc_coord
    #             dist, dist_xyz = computeVertexDist(closest_pc_coord, closest_input_coord)
    #             return closest_pc_coord, dist
    # else:
    #     return None, sys.maxsize
    return min_pc_vert, min_dist



    # for ds in pyramid_vertices:
    #     mesh_vertices_ds = downsampleVertices(mesh_vertices, ds)
    #     print(mesh_vertices_ds)
    #     print(mesh_vertices_ds & pyramid_vertices[ds])
    #     if len(mesh_vertices_ds & pyramid_vertices[ds]):
    #         min_ds = 32*ds[0]




def getClosestVertexPyramidFromPoint(
        input_coord, pc_vertices,
        pc_vertices_ds_cache,
        pc_vertices_ds_reverse_cache,
        ds_factors,
        mesh_voxel_size,
        cutoff=0):
    min_ds = (sys.maxsize, sys.maxsize, sys.maxsize)
    if len(pc_vertices) == 0:
        return None, sys.maxsize 
    min_dist = sys.maxsize
    min_pc_vert = None

    # # test if point is in block
    # to_super_block_index = getSuperBlockIndex(pc_vertices[0])
    # from_super_block_index = getSuperBlockIndex(input_coord)

    for ds in ds_factors:
        if ds[0] > min_ds[0]:
            continue
        if ds not in pc_vertices_ds_cache:
            pc_vertices_ds_cache[ds] = downsampleVertices(pc_vertices, ds, mesh_voxel_size, pc_vertices_ds_reverse_cache)
        input_coord_ds = downsampleVertex(input_coord, ds, mesh_voxel_size)
        # print(f'ds: {ds}')
        # print(f'input_coord_ds: {input_coord_ds}')
        # print(f'pc_vertices_ds_cache: {pc_vertices_ds_cache[ds]}')
        if input_coord_ds in pc_vertices_ds_cache[ds]:
            closest_pc_vertices = pc_vertices_ds_reverse_cache[input_coord_ds]
            for pc_vert in closest_pc_vertices:
                dist, dist_xyz = computeVertexDist(pc_vert, input_coord)
                # print(pc_vert)
                # print(f'pc_vert: {pc_vert}: {dist}')
                if min_dist > dist:
                    min_dist = dist
                    min_pc_vert = pc_vert
                if dist <= cutoff:
                    return min_pc_vert, min_dist
    # asdf
    return min_pc_vert, min_dist

