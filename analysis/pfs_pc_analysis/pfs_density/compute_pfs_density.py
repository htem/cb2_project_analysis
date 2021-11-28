from collections import defaultdict
from daisy import Coordinate
import compress_pickle

db = defaultdict(dict)
f = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/pf_density_proofread_210305'
with open(f) as fin:
    for line in fin:
        line = line.strip()
        if len(line) == 0:
            continue
        if line[0] == 'x':
            current_block = line
            db[current_block]['coords'] = []
            db[current_block]['pfs'] = []
        elif line[0].isdigit():
            db[current_block]['coords'].append(line)
        elif line[0] == 'p':
            pfs = line.split()
            db[current_block]['pfs'].extend(pfs)

# print(db)
def to_coordinate(xyz_str):
    xyz_str = xyz_str.split(', ')
    xyz = Coordinate(xyz_str)
    xyz = xyz * (4, 4, 40)
    return xyz

def calc_area(xyzs):
    delta = xyzs[1] - xyzs[0]
    return (delta[0]*delta[1])


total_area = 0.0
total_pfs = 0
densities = []

for block in db:
    if len(db[block]['pfs']) == 0:
        continue
    x, y = block[1:].split('y')
    db[block]['x'] = int(x)
    db[block]['y'] = int(y)
    # y_dist = 94*4000 - db[block]['y']*4000
    y_dist = 94*4 - db[block]['y']*4
    db[block]['y_dist'] = int(y_dist)

    db[block]['coords'][0] = to_coordinate(db[block]['coords'][0])
    db[block]['coords'][1] = to_coordinate(db[block]['coords'][1])

    area = calc_area(db[block]['coords']) / 1000000
    total_area += area
    num_pfs = len(db[block]['pfs'])
    total_pfs += num_pfs
    density = num_pfs / area
    db[block]['area'] = area
    db[block]['density'] = density
    # total_samples += 1
    densities.append(density)
    print(f'xy: {x} {y}')
    print(f'y_dist: {y_dist}')
    print(f'area: {area}')
    print(f'density: {density}')
    print()

print(f'average density: {total_pfs/total_area}')
print(f'average density: {sum(densities)/len(densities)}')
    # db[block]['area'] = 

compress_pickle.dump(
    db,
    '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/pfs_density_db_210306.gz')


