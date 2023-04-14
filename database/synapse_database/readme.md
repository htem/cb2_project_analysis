# Exporting synapse database

Synapses are often predicted by xyz locations and from these locations we can map them to specific segmented neurons. In MD-Seg however we have block-wise super-fragments instead of fully segmented neurons. Thus there are two databases to export from MongoDB:

1. Synapse properties by `synapse_id`s
    - Synapses are extracted from synaptic blob prediction as described in Methods.
    - Properties include xyz locations and pre/post-syn super-fragment IDs.

2. Super-fragments -> mapped `synapse_id`s
    - Map of super-fragments to synapses associated with the super-fragments.

`segway.graph` can then use these two tables, in conjunction with an "object -> super-fragments" table to construct a neuron to neuron graph.

## Downloads

Download the following files to `./data/`:
- synapse properties: download [pt0](https://github.com/htem/cb2_project_analysis_files/releases/download/230414/synapses.db.tar.gz.aa), [pt1](https://github.com/htem/cb2_project_analysis_files/releases/download/230414/synapses.db.tar.gz.ab), [pt2](https://github.com/htem/cb2_project_analysis_files/releases/download/230414/synapses.db.tar.gz.ac), [pt3](https://github.com/htem/cb2_project_analysis_files/releases/download/230414/synapses.db.tar.gz.ad)
    - After downloading all parts, run `cat synapses.db.tar.gz.* | tar xzvf -` to extract.
- super-fragments -> synapses map: [download](https://github.com/htem/cb2_project_analysis_files/releases/download/230414/superfragments.db.gz)
