import sys
import json
sys.path.insert(0, "/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/catpy")

import catpy
from catpy.applications import CatmaidClientApplication
from catpy.applications.export import ExportWidget

# import catpy
# from catpy.applications.export import ExportWidget

# client = catpy.CatmaidClient(
#     'http://catmaid3.hms.harvard.edu/catmaidcb2/',
#     '1342ab7fa3426d67890b42e4bac40c644adaf2ec'
# )

# query = {'object_ids': 42}

#annotations = client.fetch('18/annotations/query', method='POST', data=query)

# print(annotations)



class AnnotationFetcher(CatmaidClientApplication):

    def fetch_all_skeletons(self):
        # https://catmaid.readthedocs.io/en/2018.11.09/api-doc.html#operation---project_id--skeletons--get
        return self.get((self.project_id, 'skeletons'), {})


client = catpy.CatmaidClient(
    'http://catmaid3.hms.harvard.edu/catmaidcb2/',
    '1342ab7fa3426d67890b42e4bac40c644adaf2ec'
)

projects = {
    'synapse_cutout1_skeleton': 18,
    'synapse_cutout2_skeleton': 16,
    'synapse_cutout3_skeleton': 19,
    'synapse_cutout4_skeleton': 14,
    'synapse_cutout5_skeleton': 15,
    'synapse_cutout6_skeleton': 17,
    'synapse_cutout7_skeleton': 21,
    'synapse_cutout8_skeleton': 22,
    'synapse_cutout9_skeleton': 20,
    'ml0_skeleton': 24,
    'ml1_skeleton': 24,
    'pl2_skeleton': 24,
}

pids = [pid for pid in projects]
if len(sys.argv) > 1:
    pids = sys.argv[1:]

for project in pids:
    pid = projects[project]
    print("Fetching skeletons for %s" % project)

    client.project_id = pid

    annotation_fetcher = AnnotationFetcher(client)
    skeletons = annotation_fetcher.fetch_all_skeletons()
    # skeletons = [s for s in skeletons]
    # print(skeletons)
    export_widget = ExportWidget(client)
    geometry = export_widget.get_treenode_and_connector_geometry(*skeletons)

    with open('%s.json' % project, 'w') as f:
        json.dump(geometry, f)
    # print(geometry)

# for s in annotations:
#     skeleton = export_widget.get_treenode_and_connector_geometry(s)
#     print(skeleton)
