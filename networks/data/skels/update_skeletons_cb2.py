import sys
import json
sys.path.insert(0, "/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/catpy")

import catpy
from catpy.applications import CatmaidClientApplication
from catpy.applications.export import ExportWidget

def get_coordinates(offset, shape, context):

    return ((
                (offset[0]-context[0])*4,
                (offset[1]-context[1])*4,
                (offset[2]-context[2])*40,
            ),
            (
                (offset[0]+shape[0]+context[0])*4,
                (offset[1]+shape[1]+context[1])*4,
                (offset[2]+shape[2]+context[2])*40,
            ),
        )

def voxel_to_world(coord):
    return (coord[0]*4, coord[1]*4, coord[2]*40)

def subtract_offset(geometry, offset, context):

    offset = voxel_to_world(offset)
    context = voxel_to_world(context)

    for skid in geometry["skeletons"]:
        for tid in geometry["skeletons"][skid]["treenodes"]:
            for i in range(len(offset)):
                geometry["skeletons"][skid]["treenodes"][tid]["location"][i] -= offset[i]
                #geometry["skeletons"][skid]["treenodes"][tid]["location"][i] += context[i]
                if i == 2:
                   geometry["skeletons"][skid]["treenodes"][tid]["location"][i] += context[i]

    return geometry

class AnnotationFetcher(CatmaidClientApplication):

    def fetch_all_skeletons(self):
        # https://catmaid.readthedocs.io/en/2018.11.09/api-doc.html#operation---project_id--skeletons--get
        return self.get((self.project_id, 'skeletons'), {})

    def fetch_skeletons_in_bounding_box(self, offset, shape, context):
        '''offset and shape are in pixel'''

        print(get_coordinates(offset, shape, context))
        ((minx, miny, minz), (maxx, maxy, maxz)) = get_coordinates(offset, shape, context)

        # https://catmaid.readthedocs.io/en/stable/api-doc.html#operation---project_id--skeletons-in-bounding-box-get
        return self.get((self.project_id, 'skeletons', 'in-bounding-box'),
                        {
                           'minx': minx,
                           'miny': miny,
                           'minz': minz,
                           'maxx': maxx,
                           'maxy': maxy,
                           'maxz': maxz
                           })


client = catpy.CatmaidClient(
    'http://catmaid3.hms.harvard.edu/catmaidcb2/',
    '1342ab7fa3426d67890b42e4bac40c644adaf2ec'
)

# coordinates are in pixels
# 1536, 1536, 150 = 6um x 6um x 6um in TEM

# FORMAT
# GT alias: (proj_id, ((offset, shape, context)))
# WARNING: offset needs to be exactly that of the segmentation volume
projects = {
    'ml3': (2, ((100352, 62464, 182), (1536*2, 1536*2, 150), (512, 512, 50))),
    'ml5': (2, ((1024*119, 1024*67, 70), (1536*2, 1536*2, 150), (512, 512, 50))),
    'ml4_gt_drasti': (2, ((1024*112, 1024*93, 106), (1536*2, 1536*2, 150), (512, 512, 50))),
    'ml4_gt_mj': (2,((1024*160, 1024*119, 162), (1536*2, 1536*2, 150), (512, 512, 50))),
    'gcl3_jr320': (2, ((1024*100, 1024*105, 231), (1536*2, 1536*2, 150), (512, 512, 50)))
    }

pids = [pid for pid in projects]
if len(sys.argv) > 1:
    pids = sys.argv[1:]

for project in pids:
    pid, (offset, shape, context) = projects[project]
    print("Fetching skeletons for %s" % project)

    client.project_id = pid

    annotation_fetcher = AnnotationFetcher(client)
    skeletons = annotation_fetcher.fetch_skeletons_in_bounding_box(offset, shape, context)
    # skeletons = [s for s in skeletons]
    print(skeletons)
    export_widget = ExportWidget(client)
    geometry = export_widget.get_treenode_and_connector_geometry(*skeletons)

    geometry = subtract_offset(geometry, offset, context)

    with open('%s_skeleton.json' % project, 'w') as f:
        json.dump(geometry, f)



