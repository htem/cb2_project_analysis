
## Requirements

- `pip install -U git+https://github.com/htem/segway.utils#egg=segway.utils`
- `pip install -U git+https://github.com/htem/segway.mdseg#egg=segway.mdseg`

## Neuron viewer
You can use `make_static_view.py` to view proofread neurons.

- Example Granule cells:
    - `python make_static_view.py grc_253,grc_270,grc_276,grc_292,grc_304,grc_310,grc_312`
- Example Purkinje cell:
    - `python make_static_view.py pc_1`

### Tips
- We can tune viewing position:
    - `python make_static_view.py grc_253,grc_270,grc_276,grc_292,grc_304,grc_310,grc_312 --crossSectionScale 10 --projectionScale 60000 --position 95689,103170,164`
- These config files have `neuron_id`s of proofread neurons:
    - https://github.com/htem/cb2_project_analysis/tree/main/graphs/configs
