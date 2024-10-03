
## Requirements

Install MDSeg & additional packages

```
pip install -U git+https://github.com/htem/segway.utils#egg=segway.utils git+https://github.com/htem/segway.mdseg#egg=segway.mdseg
pip install jsmin requests
```

### Download the database

We need to download a copy of the exported proofread neuron database.
- See https://github.com/htem/cb2_project_analysis/tree/main/database for up-to-date instructions. Below is a snippet for convenience:

```
# assuming the current directory is `cb2_project_analysis/mdseg/`
cd ../database/data
wget https://github.com/htem/cb2_project_analysis_files/releases/download/230414/neurons_grc_mf_pc_230414.db.tar.gz
tar xf neurons_grc_mf_pc_230414.db.tar.gz
```

## Neuron viewer
You can use `make_static_view.py` to view proofread neurons.

The below assumes the current directory is `cb2_project_analysis/mdseg/`

- Example Granule cells:
    - `python make_static_view.py grc_253,grc_270,grc_276,grc_292,grc_304,grc_310,grc_312`
- Example Purkinje cell:
    - `python make_static_view.py pc_1`

### Tips
- We can tune viewing position:
    - `python make_static_view.py grc_253,grc_270,grc_276,grc_292,grc_304,grc_310,grc_312 --crossSectionScale 10 --projectionScale 60000 --position 95689,103170,164`
- These config files have `neuron_id`s of proofread neurons:
    - https://github.com/htem/cb2_project_analysis/tree/main/graphs/configs
- If there's an error with outdated GitHub token, try to make a new one by visiting https://github.com/settings/tokens/new. You would need a GitHub account and remember to select `gist` in the permissions.
    - Then run the command with `--token YOUR_TOKEN` appended
    - To see existing tokens, visit https://github.com/settings/tokens
