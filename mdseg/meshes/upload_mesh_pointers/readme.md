## Upload mesh pointers

Neuroglancer legacy precomputed mesh format requires that each object $id has a pointer file $id:0 that points to the actual mesh file. This script adds the pointer files to a hierarchical format and upload them directly to a cloud bucket.

Features:
- Options to add points to the folder or to the cloud (or both).
- Use multiprocessing to speed up cloud uploads.

TODO:
- Refactor code
