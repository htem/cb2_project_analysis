import os
# import sqlite3
# import pickle
# import numpy as np
# from threading import Lock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import tempfile

# Arguments
# in_path = '/n/f810/htem/Segmentation/cb2_v4/output.zarr/meshes/precomputed_v2/mesh'
in_path = '/n/f810/htem/Segmentation/cb2_v4/output.zarr/meshes/precomputed_v2/mesh_legacy_test'
cloud_path = 's3://upload-bossdb-6uee415thefq89jb/mesh_upload'
write_to_dir = False
# write_to_dir = True
write_to_cloud = False
write_to_cloud = True
hierarchy_size = 10000
chunk_size = 10000
# chunk_size = 1000
num_workers = 10

def compute_mesh_id(fname):
    mesh_id = 0
    levels = int(fname.split('/')[0])
    n = 0
    # print(fname)
    for num in fname.split('/')[1:]:
        mesh_id *= hierarchy_size
        num = int(num)
        assert num < hierarchy_size
        mesh_id += num
        n += 1
    assert (n-1) == levels
    return mesh_id

def make_pointer_file(mesh_id, mesh_file, tmpdir):
    fout_path = f'{tmpdir}/{mesh_id}:0'
    # print(f'writing to {fout_path}')
    with open(fout_path, 'w') as fout:
        fout.write(f'{{"fragments":["{mesh_file}"]}}')

def fn(mesh_files):
    assert len(mesh_files)
    assert len(mesh_files) <= chunk_size
    with tempfile.TemporaryDirectory() as tmpdir:
        # print(f'Creating {tmpdir}')
        for mesh_file in mesh_files:
            mesh_id = compute_mesh_id(mesh_file)
            make_pointer_file(mesh_id, mesh_file, tmpdir)
        if write_to_dir:
            os.system(f"rsync -r {tmpdir}/* {in_path}")
        if write_to_cloud:
            command = f"aws s3 cp {tmpdir} {cloud_path} --recursive --quiet"
            print(f'Running {command}')
            os.system(command)


futures = []
i = 0
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    mesh_files = []
    for dirpath, dirnames, filenames in os.walk(in_path, followlinks=True):
        if len(dirnames):
            # print(f'Skipping {dirpath}')
            continue  # leaf dirs should be entirely composed of files
        obj_path = dirpath[len(in_path)+1:]
        for f in filenames:
            mesh_files.append(f'{obj_path}/{f}')
            if len(mesh_files) == chunk_size:
                futures.append(executor.submit(fn, mesh_files))
                # fn(mesh_files)
                mesh_files = []
                i += 1
                if i > 10:
                    if (i % 10 == 0):
                        print(f'Processed {i*chunk_size}')
        # if i > 0:
        #     break
    if len(mesh_files):
        print(mesh_files)
        futures.append(executor.submit(fn, mesh_files))

for f in futures:
    f.result()
