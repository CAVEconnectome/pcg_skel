---
title: Changelog
---

`pcg_skel` aims to follow semantic versioning, such that major versions are potentially backward-incompatible, minor versions add new features, and patch versions are bug fixes. This changelog is a summary of the changes in each version.

## 1.0.0 

### Features

* The principle functions (`pcg_skeleton`, `pcg_meshwork`, and `pcg_graph`) all now use the chunkwise cache on the PCG, rather than an older approach that had to involve mesh data.
The old function names (`coord_space_*`) are now deprecated, but will work until the next major version. 
The previous functions are now available in the `pcg_skel.nocache` module.
* In `pcg_meshwork`, if synapses are requested and no synapse table is specified, the function will use the default synapse table.
* In `pcg_meshwork`, `synapses=True` will now return both pre and postsynaptic annotations.
* In `pcg_meshwork`, there is now a `synapse_point_resolution` argument that determines the resolution of the synapse points returned.
By default, this value will be `[1,1,1]` (x,y,z resolution), indicating that points should be in nanometers, the same units as the vertices.

### Changes

* In `pcg_meshwork`, when requesting synapses the partner root ids are not returned by default. `nrn.anno.pre_syn.df` will not have a `post_pt_root_id` field, and `nrn.anno.post_syn.df` will not have a `pre_pt_root_id` field.
This is to avoid confusion, because these fields can quickly become stale.
If you need them, you can still get them by using the `synapse_partners=True` argument.
Otherwise, you can use 

    ```python
    client.chunkedgraph.get_roots(nrn.anno.pre_syn.df['post_pt_supervoxel_id'], timestamp)
    ```

    to get the post-synaptic root ids for a list of supervoxels at a particular timestamp, and similar for the pre-synaptic root ids.

* In `pcg_meshwork`, the resolution of the synapse points has changed. It will now be in *nanometers*, not voxel dimensions that could change with different datasets. This means that the locations in the `nrn.anno.pre_syn.df` and `nrn.anno.post_syn.df` dataframes will already be in the same coordinates as vertices without an additional conversion step.

