# CB2 project
This code repository accompanies the publication [Structured connectivity in the cerebellum enables noise-resilient pattern separation](https://www.biorxiv.org/content/10.1101/2021.11.29.470455v1).

## Dataset browser
[Open viewer](http://catmaid2.hms.harvard.edu:33400/v/e8be44a62ac7a87889cd5d8d56c80248c0a8c63f/#!%7B%22dimensions%22:%7B%22x%22:%5B4e-09,%22m%22%5D,%22y%22:%5B4e-09,%22m%22%5D,%22z%22:%5B4e-08,%22m%22%5D%7D,%22layers%22:%5B%7B%22type%22:%22image%22,%22source%22:%22python://volume/e8be44a62ac7a87889cd5d8d56c80248c0a8c63f.ee3800de2c117f2ae81586cc20ccc78f30eda941%22,%22name%22:%22raw%22%7D%5D,%22position%22:%5B135280.0,102120.0,312.0%5D,%22crossSectionScale%22:18.0,%22layout%22:%22xy%22%7D)

## Example mesh reconstructions

These are mesh renderings that have been exported (as lists of segments) from our proofreading platform's database (MDSeg).
Unfortunately at this point the infrastructure is not yet capable of allowing randomly accessing neurons, though this may be added in the future.

### Granule cells
* Two adjacent granule cells: [viewer](http://catmaid2.hms.harvard.edu:33401/v/two_grcs/)
* A few granule cells: [viewer](http://catmaid2.hms.harvard.edu:33401/v/grcs0/)

![grcs](constructions/resources/grcs.png)

### Purkinje cells
**WARNING: These neurons are huge. Make sure you have free RAM (4GB+) before loading.**
* "purkinje_0" cell: [viewer](http://catmaid2.hms.harvard.edu:33401/v/purkinje_0/)
* "pc_9" cell: [viewer](http://catmaid2.hms.harvard.edu:33401/v/pc_9/)
* "pc_15" cell: [viewer](http://catmaid2.hms.harvard.edu:33401/v/pc_15/)

![pc_15](constructions/resources/pc_15.jpg)

## Analysis source code
See README.md in `analysis/` for more information.

## Related software
* [Daisy](https://github.com/funkelab/daisy/): a blockwise task scheduler for processing large nD datasets.
* MDSeg: a merge-deferred, targeted proofreading platform based on Neuroglancer
  * [Modified Neuroglancer front-end](https://github.com/htem/neuroglancer_pr/tree/segway_pr_v2)
  * [Python back-end](https://github.com/htem/segway.mdseg)

## License
This software is licensed under the [MIT](LICENSE) license.
