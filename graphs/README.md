# NetworkX graphs of MF-GrC and GrC-PC connectivity

## Glossary
- `SynapseGraph`: internal data structure that is used for integrating segmentation data (at the level of super-voxel), synapse prediction data, and proofreading data.
    - See https://github.com/htem/segway.graph
- `syndb`: legacy data structure used for synapse filtering and some analyses.
- `input_graph`: binary connectivity data structure used for analyses and activity simulation.
- `NetworkX`: portable connectivity graphs (https://networkx.org/)

## History & Aims

The first release of this repo interminglely uses `SynapseGraph`, `syndb`, `input_graph` for different analyses, which can be confusing and difficult to use for new (and old) users. Here we collapse the different formats to NetworkX so that research dissemination is easier. With this potential users will only need to install and learn the syntax of `NetworkX` to analyze the provided graphs.

For de-novo datasets, instead of the legacy flow `SynapseGraph` -> `syndb` -> `input_graph`, we will/should simply just do `SynapseGraph` -> `NetworkX`.

## MF-GrC graphs

TLDR: what you're looking for is probably the binary connectivity graph of MF-GrC, which can be downloaded at: https://github.com/htem/cb2_project_analysis_files/releases/download/230103/graph_mf_grc_binary_210519.gz. This graph was used for both connectivity analyses as well as simulations.

For an example usage, see the Jupyter notebooks in [`./notebooks`](./notebooks).

### Levels of available graphs

There are several levels of representation of MF-GrC connectivity:
- With all synapses, extracted as-is (`all`)
- With nearby, possibly duplicated synapses coalesced (`coalesced`)
    - See https://github.com/htem/cb2_project_analysis/blob/main/analysis/gen_db/coalesce2.py
        - This computes `all` -> `coalesced`
- With synapses between pairs of neurons collapsed to a single binary connection (`binary`)
    - See, e.g., https://github.com/htem/cb2_project_analysis/blob/main/analysis/gen_db/mf_grc/gen_input_graph_210520_all.py
        - This computes `coalesced` -> `input_graph`
    - We used primarily used this graph for connectivity analysis and dimensionality and activity simulation
- Replicated mf-grc binary graphs (`replicated`)
    - `binary` graph is replicated along the z-axis with connectivity re-randomized
    - Usage: large-scale dimensionality and activity simulation
        - Simulation scripts loads `input_graph`, and in memory randomizes the graph

### NetworkX files

These files were converted from legacy formats using the scripts (`convert_syndb_to_networkx_mf_grc.py`, `convert_input_graph_to_networkx.py`) in this folder:

- `all`: [210518](https://github.com/htem/cb2_project_analysis_files/releases/download/230103/graph_mf_grc_synapse_210518_all.gz)
- `coalesced`: [210518](https://github.com/htem/cb2_project_analysis_files/releases/download/230103/graph_mf_grc_synapse_210518_coalesced.gz)
- `binary`: [210518](https://github.com/htem/cb2_project_analysis_files/releases/download/230103/graph_mf_grc_binary_210519.gz)

See [`notebooks/test_mf_grc_networkx.ipynb`](./notebooks/test_mf_grc_networkx.ipynb) for accessing these files.

### `SynapseGraph` files

- [230301](https://github.com/htem/cb2_project_analysis_files/releases/download/230103/synapsegraph_mf_grc_230301.npz)
    - One can generate a NetworkX graph directly with the built-in `make_networkx_graph()`
    - See [`make_grc_mf_graph.py`](./make_grc_mf_graph.py)

### TODOs

- [ ] `replicated`:
    - [ ] Pull out the replication function from `input_graph` and generate a few example graphs.
- [ ] NetworkX-based scripts
    - [ ] Coalesce synapses (`all` -> `coalesced`)
        - This is a bit hard because the old coalescing script uses not only location proximity but also the underlying segmentation to determine if two synapses should be merged or not.
    - [ ] Extract binary connectivity (`boutoned` -> `binary`)
        - TODOs include using DBSCAN to extract the bouton locations out of single MF axons and using additional parameters (e.g., ignoring single synapse connections which are likely mispredictions) and proofreading data to extract a high quality binary graph.
    - [ ] Replicated binary graph (`binary` -> `replicated`)
