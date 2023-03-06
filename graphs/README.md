# NetworkX graphs of MF-GrC and GrC-PC connectivity

The first release of this repo interminglely uses `SynapseGraph`, `syndb`, `input_graph` for different analyses, which can be difficult for users to use and understand. Here we collapse the different formats to NetworkX to make data dissemination easier. With this potential users will only need to install and learn the syntax of `NetworkX` to analyze the provided graphs.

## TLDR
- **For MF-GrC connectivity**
    - What you're looking for is probably the binary connectivity graph of MF-GrC ([graph_mf_grc_binary_210519.gz](https://github.com/htem/cb2_project_analysis_files/releases/download/230103/graph_mf_grc_binary_210519.gz)). This graph was used for both connectivity analyses as well as simulations.
    - For an example usages, see the Jupyter notebooks in [`notebooks/`](./notebooks):
        - [test_mf_grc_networkx.ipynb](./notebooks/test_mf_grc_networkx.ipynb)
        - [analysis_grc_input_sharing.ipynb](./notebooks/analysis_grc_input_sharing.ipynb)
- **For GrC-PC connectivity**
    - The "local" GrC graph and "remote" pfs graph are stored and processed separately
    - Local GrCs: [graph_grc_pc_synapse_210429_coalesced.gz](https://github.com/htem/cb2_project_analysis_files/releases/download/230306/graph_grc_pc_synapse_210429_coalesced.gz)
    - Remote pfs: [graph_pfs_pc_synapse_210429_coalesced.gz](https://github.com/htem/cb2_project_analysis_files/releases/download/230306/graph_pfs_pc_synapse_210429_coalesced.gz)
    - Example notebook: [analysis_comparing_local_and_remote_pfs.ipynb](./notebooks/analysis_comparing_local_and_remote_pfs.ipynb)

## Glossary

- `SynapseGraph`: internal data structure that is used for integrating segmentation data (at the level of super-voxel), synapse prediction data, and proofreading data.
    - See https://github.com/htem/segway.graph
- `syndb`: legacy data structure used for synapse filtering and some analyses.
- `input_graph`: binary connectivity data structure used for analyses and activity simulation.
- `NetworkX`: portable connectivity graphs (https://networkx.org/)

## MF->GrC

### Levels of legacy files

There are several levels of representation of MF-GrC connectivity:
- With all synapses, extracted as-is (`all`)
- With nearby, possibly duplicated synapses coalesced (`coalesced`)
    - See the legacy [coalesce2.py](/analysis/gen_db/coalesce2.py) script.
        - This computes `all` -> `coalesced`
- With synapses between pairs of neurons collapsed to a single binary connection (`binary`)
    - See, e.g., [gen_input_graph_210520_all.py](/analysis/gen_db/mf_grc/gen_input_graph_210520_all.py)
        - This computes `coalesced` -> `input_graph`
        - Will also work with `all` -> `input_graph`; synapse coalescing is not necessary.
    - We used primarily used this graph for connectivity analysis and dimensionality and activity simulation
- Replicated mf-grc binary graphs (`replicated`)
    - The `binary` graph is replicated along the z-axis with connectivity re-randomized
    - Usage: large-scale dimensionality and activity simulation
        - Simulation scripts loads `input_graph`, and in memory randomizes the graph


### `SynapseGraph` files

- [230301](https://github.com/htem/cb2_project_analysis_files/releases/download/230103/synapsegraph_mf_grc_230301.npz)
    - One can generate a NetworkX graph directly with the built-in `make_networkx_graph()`
    - See [`make_grc_mf_graph.py`](./make_grc_mf_graph.py)
        - [ ] TODO: make `SynapseGraph` be able to read from binary files besides the internal MongoDB servers.

### NetworkX files

These files were converted from legacy formats using the scripts (`convert_syndb_to_networkx_mf_grc.py`, `convert_input_graph_to_networkx.py`) in this folder:

- `all`: [210518](https://github.com/htem/cb2_project_analysis_files/releases/download/230103/graph_mf_grc_synapse_210518_all.gz)
- `coalesced`: [210518](https://github.com/htem/cb2_project_analysis_files/releases/download/230103/graph_mf_grc_synapse_210518_coalesced.gz)
- `binary`: [210518](https://github.com/htem/cb2_project_analysis_files/releases/download/230103/graph_mf_grc_binary_210519.gz)

See [`notebooks/test_mf_grc_networkx.ipynb`](./notebooks/test_mf_grc_networkx.ipynb) for accessing these files.

### TODOs

- [ ] `replicated`:
    - [ ] Pull out the replication function from `input_graph` and generate a few example graphs.
- [ ] NetworkX-based scripts
    - [ ] Coalesce synapses (`all` -> `coalesced`)
        - This is a bit hard because the old coalescing script uses not only location proximity but also the underlying segmentation to determine if two synapses should be merged or not.
    - [ ] Extract binary connectivity (`boutoned` -> `binary`)
        - TODOs include using DBSCAN to extract the bouton locations out of single MF axons and using additional parameters (e.g., ignoring single synapse connections which are likely mispredictions) and proofreading data to extract a high quality binary graph.
    - [ ] Replicated binary graph (`binary` -> `replicated`)

## GrC->PC

### Levels of legacy files

Like with `MF->GrC`, we also had `all`, `coalesced`, and `binary`.

### `SynapseGraph` files

To be done.

### NetworkX files

These files were converted from legacy formats using the script `convert_syndb_to_networkx_grc_pc.py` in this folder:
- Local GrCs:
    - `all`: [210429](https://github.com/htem/cb2_project_analysis_files/releases/download/230306/graph_grc_pc_synapse_210429_all.gz)
    - `coalesced`: [210429](https://github.com/htem/cb2_project_analysis_files/releases/download/230306/graph_grc_pc_synapse_210429_coalesced.gz)
- Remote pfs:
    - `all`: [210429](https://github.com/htem/cb2_project_analysis_files/releases/download/230306/graph_pfs_pc_synapse_210429_all.gz)
    - `coalesced`: [210429](https://github.com/htem/cb2_project_analysis_files/releases/download/230306/graph_pfs_pc_synapse_210429_coalesced.gz)

See [analysis_comparing_local_and_remote_pfs.ipynb](./notebooks/analysis_comparing_local_and_remote_pfs.ipynb) for how to compute the binary graph out of the synapse graphs.
