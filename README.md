## Phantom-voxelized

See [About phantoms and projectors](https://github.com/xcist/documentation/wiki/About-phantoms-and-projectors).

These are large files, so you might want to be selective about which ones you download.

Each phantom describes a 3D volume, and includes multiple material "volume fraction maps" (.density_) and one .json file.

Some phantoms are also provided as a single slab (one voxel in Z). These are a subset of the 3D phantom of the corresponding name. Because these have a smaller Z extent, they may not include some organs included in the full 3D phantom, and therefore might have fewer material maps. Furthermore, these are more efficient in that each material map has a unique XY size and offset, because only pixels that contain that material are included. Therefore, these are convenient for fast, efficient simulations during simulation parameter selection and code development.
