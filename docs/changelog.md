---
title: Changelog
---

`pcg_skel` aims to follow semantic versioning, such that major versions are potentially backward-incompatible, minor versions add new features, and patch versions are bug fixes. This changelog is a summary of the changes in each version.

## [1.2.2]

* Fixed `chunk_tools.get_closest_lvl2_chunk` to correctly handle out-of-date root ids.

## [1.2.1]

### Fixes

* Fixed various issues with meshwork rehydration.

## [1.2.0]

### Features

* Added hydration of meshwork objects from the new-ish CAVE skeleton service.

## [1.1.0]

### Features

* Added `add_synapse_count` and `aggregate_property_to_skeleton` functions to `features` module.
* Updated tests to use the `CAVEclientMock` feature introduced in CAVEclient 6.4.0.

## [1.0.7]

### Fixes

* Fixed errors arising from in mesh-to-skel-map for single vertex root ids.


## [1.0.6]

### Fixes

* Fixed errors arising from running pcg_graph, pcg_skeleton, or pcg_meshwork on a root id with only a single vertex.

## [1.0.5]

### Features

* Adds an optional `return_quality` argument to `features.add_is_axon_annotation` to optionally return the split quality value.

## [1.0.4]

### Fixes

* Raise a clear error when a level 2 cache service, which is needed, is not available.

## [1.0.3]

### Fixes

* Fixed the `features.add_is_axon_annotation` function to warn rather than raise an error when the split quality is poor.

## [1.0.2]

### Fixes

* Fixed the dictionaries returned with `pcg_skeleton`, such that one maps skeleton vertices to l2 ids and the other maps l2 ids to skeleton vertices. Previously, the first dictionary was repeated twice.

## [1.0.1]

### Changes

* In pcg_graph, pcg_skeleton and pcg_meshwork, you can now pass the result of `client.chunkedgraph.level2_chunk_graph` directly as an argument (`level2_graph`) to avoid recomputing the graph and level2 features if you already have it around. The format is a list of pairs of level 2 ids.

## 1.0.0 

### Features

* The principle functions (`pcg_skeleton`, `pcg_meshwork`, and `pcg_graph`) all now use the chunkwise cache on the PCG, rather than an older approach that had to involve mesh data.
The old function names (`coord_space_*`) are now deprecated, but will work until the next major version. 
The previous functions are now available in the `pcg_skel.nocache` module.
* In `pcg_meshwork`, if synapses are requested and no synapse table is specified, the default synapse table will be used.
* In `pcg_meshwork`, `synapses=True` will now return both pre and postsynaptic annotations.
* In `pcg_meshwork`, there is now a `synapse_point_resolution` argument that determines the resolution of the synapse points returned.
By default, this value will be `[1,1,1]` (x,y,z resolution), indicating that points should be in nanometers, the same units as the vertices.
* In `pcg_meshwork`, there is now the option to specify the name of the pre and post synapse columns to use as representitiative points in a synapse table.
* Added `pcg_skeleton_direct` function that expects vertices and edges from an already computed L2 graph.

### Changes

* In `pcg_meshwork`, when requesting synapses the partner root ids are not returned by default. Accordingly, `nrn.anno.pre_syn.df` will not have a `post_pt_root_id` field, and `nrn.anno.post_syn.df` will not have a `pre_pt_root_id` field.
This is to avoid confusion, because these fields can quickly become stale.
If you need them, you can still get them by using the `synapse_partners=True` argument.
However, the supervoxel ids are returned, which both do not change and also let you look up root ids when needed.
Otherwise, you can use 

    ```python
    client.chunkedgraph.get_roots(nrn.anno.pre_syn.df['post_pt_supervoxel_id'], timestamp)
    ```

    to get the post-synaptic root ids for a list of supervoxels at a particular timestamp, and similar for the pre-synaptic root ids.

* In `pcg_meshwork`, the resolution of the synapse points has changed. It will now be in *nanometers*, not voxel dimensions that could change with different datasets. This means that the locations in the `nrn.anno.pre_syn.df` and `nrn.anno.post_syn.df` dataframes will already be in the same coordinates as vertices without an additional conversion step.

### Notes for upgrading.

1) If you are using the `pcg_skel.coord_space_*` functions, you should switch to the `pcg_*` function names.

2) If you are using synaptic partners from meshwork objects directly, you need to set the `synapse_partners=True` argument.

3) If you are using synapse points, the default resolution has changed to nanometers. If you hard-coded the old resolution, you should update your code to reflect this change. This will not affect distances measured along the arbor of the neuron that use the graph topology, only spatial properties like `ctr_pt_position`.