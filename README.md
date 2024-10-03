# CB2 project
This code repository accompanies the Nature 2023 publication [Structured cerebellar connectivity supports resilient pattern separation](https://www.nature.com/articles/s41586-022-05471-w) ([paywall-free ShareIT link](https://rdcu.be/c0hLW)).

## Updates!

- **March 6 2023**: we added NetworkX graph files and examples to [graphs](./graphs). Using NetworkX is easier and more portable than previous legacy formats.

## Dataset browser
[Open viewer](http://catmaid2.hms.harvard.edu:33400/v/e8be44a62ac7a87889cd5d8d56c80248c0a8c63f/#!%7B%22dimensions%22:%7B%22x%22:%5B4e-09,%22m%22%5D,%22y%22:%5B4e-09,%22m%22%5D,%22z%22:%5B4e-08,%22m%22%5D%7D,%22layers%22:%5B%7B%22type%22:%22image%22,%22source%22:%22python://volume/e8be44a62ac7a87889cd5d8d56c80248c0a8c63f.ee3800de2c117f2ae81586cc20ccc78f30eda941%22,%22name%22:%22raw%22%7D%5D,%22position%22:%5B135280.0,102120.0,312.0%5D,%22crossSectionScale%22:18.0,%22layout%22:%22xy%22%7D)

## Example mesh reconstructions

These are mesh renderings that have been exported (as lists of segments) from our proofreading platform's database (MDSeg).

To visualize other proofread neurons, see the directions in [mdseg](./mdseg).

### Granule cells
* A few granule cells: [viewer](https://htem.github.io/neuroglancer-mdseg/#!https://gist.githubusercontent.com/trivoldus28/263bdc615a2baa985dc9cb4a92981d47/raw/4abb71d4bd52fb20930787818588112d9fd5fea5/state.json)

![grcs](constructions/resources/grcs.png)

### Purkinje cells
**WARNING: These neurons are huge. Make sure you have free RAM (4GB+) before loading, and it will take a long time to load the meshes**
* "purkinje_0" cell: [viewer](https://htem.github.io/neuroglancer-mdseg/#!https://gist.githubusercontent.com/trivoldus28/25e4b83abbfbb05f504f36e81127895c/raw/70543bfe862647f9ed6ca5d72f82b5e6f8bb3e0b/state.json)

![pc_15](constructions/resources/pc_15.jpg)

## Analysis source code
See [analysis](./analysis) for more information.

## Downloadable NetworkX graphs
See [graphs](./graphs) for more information.

## Related software
* [Daisy](https://github.com/funkelab/daisy/): a blockwise task scheduler for processing large nD datasets.
* MDSeg: a merge-deferred, targeted proofreading platform based on Neuroglancer
  * [Modified Neuroglancer front-end](https://github.com/htem/neuroglancer_pr/tree/segway_pr_v2)
  * [Python back-end](https://github.com/htem/segway.mdseg)

## License
This software is licensed under the [MIT license](./LICENSE).
