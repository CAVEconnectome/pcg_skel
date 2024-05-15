---
title: Background 
---

## Background

Skeletonization is an essential part of measuring neuronal anatomy, but in 3d segmented data it is not always trivial to produce a skeleton.
Segmentations and meshes are often prohibitively large to download *en mass* and can have artifacts that generate ambiguities or make attaching annotations like synapses to specific locations unclear.

PCG-skel uses the same chunking that allows the [PyChunkedGraph](https://github.com/seung-lab/PyChunkedGraph) (PCG) to quickly make edits to large, complex neuronal segmentations and, combined with a dynamic caching system that updates after every edit, can generate complete representations of the topology of objects in just a few seconds and minimal data downloads.
Becuase the data is always matched to the underlying representation of the segmentation, there are no ambiguities in what parts are connected to what other or in which vertex a synapse or other annotation is associated with. Light data needs and rapid skeletonization make it useful in environments where analysis is being re-run on frequently changing cells.

However, there is a trade-off in terms of resolution.
The dependency on chunk size means that vertices are roughly a chunk width apart, which in current datasets like Microns amounts to about 2 microns.
Thus for understanding the overall structure of a cell or looking at long distance relationships between points along an arbor, these skeletons are quite good, but for detailed analysis at short length scales (0-10 microns or so) where being plus or minus a micron would hurt analysis, we recommend looking at other approaches like kimimaro, meshteasar, or CGAL skeletonization.

## Key terms

Ids in the PCG combine information about chunk level, spatial location, and unique object id.
This package uses the highest-resolution chunking, level 2, to derive neuronal topology and approximate spatial extent.
For clarity, it's useful to define a few terms:

* *Level 2 chunk*: A box defining a unit of data storage and representation. The entire dataset is tiled by nonoverlapping chunks. Each chunk has properties like a detailed graph of which supervoxels touch what other supervoxels, meshes associated with each segmented object inside the chunk, etc. By chunking the data and agglomerating larger objects out of these chunks, edits only have to touch those few chunks that actually change during proofreading, reducing the amount of memory and effort needed to process them.

Chunks can exist at many scales, but "level 2" refers to the lowest level of chunking (level 1 refers to the supervoxels themselves).

* *Level 2 id (L2 id)*: A segmentation id that describes the state of the segmentation inside a given L2 chunk.
Note that if two distinct parts of the same neuron enter the same chunk, each has its own level 2 id.

* *Level 2 graph*: Each level 2 id can be thought of as connected to level 2 ids in other chunks when an object's supervoxels run across chunk boundaries or where edges have introduced by merges during proofreading. The graph of which level 2 ids are connected to which others is called the "level 2 graph." Keeping track of the level 2 graph is one of the jobs of the PCG.

* *Level 2 skeleton*: A reduced version of the level 2 graph that is tree-like.

* *Representitative point*: For each level 2 id, the L2 Cache determines a representative point that is guaranteed to be within the segmentation and is located at a "most central" point in the segmentation of the L2 id.


    