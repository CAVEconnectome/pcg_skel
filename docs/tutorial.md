---
title: Tutorial
---


## Before using pcg_skel

Before you can use pcg_skel, you need to have a functioning [CAVEclient](https://caveconnectome.github.io/CAVEclient/) setup, including token, for your dataset.
This tutorial will use the `minnie65_public` dataset as an example, which can be set up by following the instructions at [this link](https://allenswdb.github.io/microns-em/em-caveclient-setup.html).

## Generating a skeleton

To get a skeleton requires only a root id and a caveclient, but there are many options to customize the skeletonization process.

At its most basic, we are going to get a skeleton for a single neuron in the `minnie65_public` dataset and the v795 data release.

```python
import caveclient
import pcg_skel

datastack = 'minnie65_public'
client = caveclient.CAVEclient(datastack)
client.materialize.version = 795 # Ensure we will always use this data release

root_id = 864691135397503777
skel = pcg_skel.pcg_skeleton(root_id=root_id, client=client)
```

The above code will generate a skeleton for [this neuron](https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/5553224754921472) (Link to Neuroglancer, needs the same credentials as above).

You can, for example, check the overall path length of the cell with `skel.path_length()` and find more features in the [Meshparty documentation](https://meshparty.readthedocs.io/en/latest/).

However, this may not be the skeleton you want most. For example, the location of the root node will be a random end point and thus the orientation of the skeleton will be arbitrary.

To control the location of the root node and the behavior around the cell body, there are several key parameters:

* *root_point*, *root_point_resolution*: Setting a root point defines the root of the skeleton, which establishes an orientation for the skeleton. Root point is an x,y,z position, and the resolution is the value in nm of the resolution of coordinates (also in x,y,z, resolution). If the point is from neuroglancer or an annotation table, take careful note of the resolution. 

* *collapse_soma*, *collapse_radius*: These two options let you collapse vertices around a soma into the root point. Setting collapse_soma to True will collapse all vertices within the collapse radius (in nanometers) into the root point. Additionally, the root point is added as a new skeleton vertex. Again, the default value works for most cortical neurons, but cells with larger or smaller cell bodies might need different values.

In addition, the skeletonization process has a minimum scale such that branches shorter than this are not included in the skeleton. This is set by the `invalidation_d` parameter, which is the distance in nanometers at which vertices are collapsed into a branch. This is an important parameter to customize for your data, since different morphologies will be best represented by different values. The default value works well for cortical neurons, for example, but is probably too coarse for fly neurons. This is controlled with:

* *invalidation_d*: The invalidation distance for skeletonization, in nanometers. This parameter sets the distance at which vertices of the graph are collapsed into a branch. Too big and branches might be missed, but too small and false branches might be added to thick processes.

There are also several adminstrative parameters, such as:

* *cv*: Passing an initialized cloudvolume object (`cv=`) can save a second or two per skeleton. This isn't a big deal for a couple of skeletons, but may save real time if you are creating a large number of skeletons.

* *return_mesh*, *return_l2dict*, *return_l2dict_mesh*: These three values are all set to False by default, which is fine if you just want a skeleton. However, if you want to map vertices to level 2 ids or skeleton vertices back to the mesh graph, these options can give you the mesh and dictionaries mapping vertices to the l2 ids for the skeletons and the mesh graph.

A more complete version of the above code might look like:

```python
import caveclient
import pcg_skel

datastack = 'minnie65_public'
client = caveclient.CAVEclient(datastack)
client.materialize.version = 795 # Ensure we will always use this data release

root_id = 864691135397503777

# Get the location of the soma from nucleus detection:
root_resolution = [1,1,1] # Cold be another resolution as well, but this will mean the location is in nm.
soma_df = client.materialize.views.nucleus_detection_lookup_v1(
    pt_root_id = root_id
    ).query(
        desired_resolution = root_resolution
    )
soma_location = soma_df['pt_position'].values[0]

# Use the above parameters in the skeletonization:

skel = pcg_skel.pcg_skeleton(
    root_id,
    client,
    root_point=soma_location,
    root_point_resolution=root_resolution,
    collapse_soma=True,
    collapse_radius=7500,
)
```

Now you can see that the root position aligns with the soma location by going to the position specified at `skel.root_position / [4,4,40]` in the Neuroglancer link above.
Note the elementwise division, because the resolution of the `minnie65_public` neuroglancer link is `[4,4,40]` nm/voxel.

## Generating a meshwork

Meshwork objects are a way to simultaneously track neuronal morphology like the skeleton as well as annotations and labels like synapses or compartments.

Most of the information needed to generate a meshwork is the same as for a skeleton. For convenience, however, synapses can be queried and added to the object by default and this comes with a few extra parameters.

Starting from where we were before, you can generate a meshwork for that same neuron with virtually the same arguments, plus `synapses=True`:

```python

nrn = pcg_skel.pcg_meshwork(
    root_id = root_id,
    client = client,
    root_point = soma_location,
    root_point_resolution = root_resolution,
    collapse_soma = True,
    collapse_radius = 7500,
    synapses=True,
)
```

All meshwork objects created this way will have an annotation table `lvl2_ids` that has the level 2 id of each vertex in the meshwork graph. This is useful for mapping synapses or other annotations back to the meshwork.

Now you can glance at some of the synapses through the `nrn.anno` attribute.
Presynaptic sites (outputs) are put under `nrn.anno.pre_syn` and postsynaptic sites (inputs) are put under `nrn.anno.post_syn`.
For example, `nrn.anno.post_syn.df.head()` will show the first few post-synaptic sites:

|    |        id |   size |   pre_pt_supervoxel_id |   post_pt_supervoxel_id |    post_pt_root_id | pre_pt_position           | post_pt_position          | ctr_pt_position           |   post_pt_level2_id |   post_pt_mesh_ind |   post_pt_mesh_ind_filt |
|---:|----------:|-------:|-----------------------:|------------------------:|-------------------:|:--------------------------|:--------------------------|:--------------------------|--------------------:|-------------------:|------------------------:|
|  0 | 172461633 |  15200 |      90359378338883207 |       90359378338887551 | 864691135397503777 | [745048. 418840. 888160.] | [744744. 419112. 888240.] | [744776. 419024. 888160.] |  162416972376572296 |               7995 |                    7995 |
|  1 | 142677506 |  22532 |      88108334439511704 |       88108265720062299 | 864691135397503777 | [678880. 441128. 890840.] | [679104. 440728. 890800.] | [679216. 440768. 890720.] |  160165859757654524 |               3412 |                    3412 |
|  2 | 145066644 |   4532 |      88036110403862351 |       88036110403865679 | 864691135397503777 | [677880. 387048. 939680.] | [677400. 387152. 939720.] | [677648. 387344. 939760.] |  160093704441299965 |               3236 |                    3236 |
|  3 | 170547369 |   2320 |      90357866644502688 |       90357866644495463 | 864691135397503777 | [743560. 374824. 925320.] | [743824. 374696. 924840.] | [743512. 374784. 925040.] |  162415460682301674 |               7940 |                    7940 |
|  4 | 149896539 |   2660 |      88811815655855809 |       88882184400027792 | 864691135397503777 | [700472. 436192. 873960.] | [700576. 436056. 874120.] | [700536. 436168. 874080.] |  160939778437546796 |               4447 |                    4447 |

Note the `mesh_ind` column, which aligns with the mesh indices in the `nrn.mesh` attribute.


## Using Features

Skeletions are only so useful on their own, one also wants to attach properties like dendritic compartments to them for analysis.
There are a few convenience functions in `pcg_skel.features` that can be used to add features to a skeleton.
All of these functions start use the level 2 chunks and collect a variety of features either from the l2cache or CAVE database.

### Synapses

The best way to get synapses is by default through the `pcg_meshwork` function, which handles the complexity of synapse queries and attachment.
Once created, these synapses can be used to generate two different mappings of the neuron.
The first creates a count of the number of synapses per skeleton vertex while dropping connectivity information, and the second uses synapses to predict which parts of the skeleton are dendritic and which are axonal.

To get the synapse count map, use the `features.add_synapse_count` function.
For example, assuming you have a pcg meshwork object `nrn` with synapses attached in the default way, we can add synapse counts to the skeleton with:

```python
pcg_skel.features.add_synapse_count(
    nrn,
)

syn_count_df = nrn.anno.synapse_count.df
```

The resulting dataframe by default will create a dataframe with one row per graph vertex (not skeleton vertex!) and columns:

* `num_syn_in`: The number of input synapse at the vertex
* `num_syn_out`: The number of output synapse at the vertex
* `net_size_in`: The summed size of the input synapses at the vertex.
* `net_size_out`: The summed size of the output synapses at the vertex.

The last two columns enable one to compute averages of synapse size, although not percentiles.

If you want to map these values to skeleton nodes, there is also a function to handle this aggregation.

```python

skel_df = pcg_skel.features.aggregate_property_to_skeleton(
    nrn,
    'synapse_count',
    agg_dict={'num_syn_in': 'sum', 'num_syn_out': 'sum', 'net_size_in': 'sum', 'net_size_out': 'sum'},
)
```

This will generate a dataframe with one row per skeleton vertex and the columns will aggregate the synapse count properties to the skeleton.
The `agg_dict` property lets you specify exactly which columns to aggregate by across associated graph vertices and how to aggregate them.
Anything that works in a pandas groupby operation will work here.
Note that if you don't specify a column in the `agg_dict`, nothing will happen to it.


