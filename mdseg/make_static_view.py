import sys
import argparse
import json
from jsmin import jsmin

import segway.mdseg.database

ng_client = 'https://htem.github.io/neuroglancer-mdseg'
raw_source = 'precomputed://s3://bossdb-open-data/nguyen_thomas2022/cb2/em'
seg_source = 'n5://https://catmaid3.hms.harvard.edu/cb2o2/staged_alignment_v3/delete_me/seg5'
mesh_source = 'precomputed://https://catmaid3.hms.harvard.edu/cb2o2/staged_alignment_v3/delete_me/mesh4'

ap = argparse.ArgumentParser()
ap.add_argument("neurons", type=str, default=None, help='Comma separated')

ap.add_argument("--colors", type=str, default=None, help='Colors for each neuron, exactly one for each. Must be hex string starting with #. Comma separated')
ap.add_argument("--neurondb_file", type=str, default='../database/data/neurons_grc_mf_pc_230414.db')
ap.add_argument("--template_file", type=str, default='./static_template_cb2.json')
ap.add_argument("--print_only", action='store_true', default=False, help='Print a pastable JSON state instead of uploading JSON')

ap.add_argument("--filename", type=str, default='state.json', help='GitHub filename')
ap.add_argument("--token", type=str, default='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', help='GitHub token')
ap.add_argument("--description", type=str, default=None, help='GitHub file description (optional)')

ap.add_argument("--position", type=str, default=None, help='Comma separated')
ap.add_argument("--crossSectionScale", type=str, default=None, help='')
ap.add_argument("--projectionScale", type=str, default=None, help='')
ap.add_argument("--colorSeed", type=int, default=None, help='')
ap.add_argument("--projectionOrientation", type=str, default=None, help='Comma separated')

args = ap.parse_args()

neuron_db = segway.mdseg.database.NeuronDBServerSQLite(args.neurondb_file)

with open(args.template_file) as f:
    view_json = json.loads(jsmin(f.read()))

view_json['layers'][0]['source'] = raw_source
view_json['layers'][1]['source'] = mesh_source
view_json['layers'][2]['source'] = seg_source

if args.position:
    args.position = args.position.split(',')
    view_json['position'] = args.position
if args.crossSectionScale:
    view_json['crossSectionScale'] = float(args.crossSectionScale)
if args.projectionScale:
    view_json['projectionScale'] = float(args.projectionScale)
if args.projectionOrientation:
    view_json['projectionOrientation'] = args.projectionOrientation.split(',')

if args.colorSeed:
    view_json['layers'][2]['colorSeed'] = args.colorSeed

segments = []
if args.neurons is not None:
    neurons = args.neurons.split(',')
    for nid in neurons:
        try:
            neuron = neuron_db.get_neuron(nid)
        except:
            raise RuntimeError(f'{nid} not found!')
        fragments = neuron.segments
        view_json['layers'][2]['equivalences'].append(fragments)
        view_json['layers'][2]['segments'].append(min(fragments))
        segments.append(min(fragments))

if args.colors is not None:
    args.colors = args.colors.split(',')
    view_json['layers'][2]['segmentColors'] = {}
    for nid, color in zip(segments, args.colors):
        assert len(color) == 7, f"Color must be 7 character long, given {color}"
        assert color[0] == '#', f"Color must begin with #, given {color}"
        for d in color[1:]:
            assert d in string.hexdigits, f"Color must be hex, given {color}"
        view_json['layers'][2]['segmentColors'][nid] = color

json_str = json.dumps(view_json)

if args.print_only:
    print(json_str)
    exit()

from segway.utils import post_gist
ret = post_gist(json_str, filename=args.filename, token=args.token, description=args.description)
url = ret['files']['state.json']['raw_url']
print(f'{ng_client}/#!{url}')


