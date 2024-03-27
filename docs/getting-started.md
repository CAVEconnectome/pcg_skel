---
title: Tutorial
---

!!! important
    Before getting started, please ensure that you have a functioning [CAVEclient](https://caveconnectome.github.io/CAVEclient/) setup, including token. If you do not have a CAVEclient setup, please follow the instructions at [this link](https://allenswdb.github.io/microns-em/em-caveclient-setup.html).

## Background

Skeletonization is an essential part of measuring neuronal anatomy, but in 3d segmented data it is not always trivial to produce a skeleton.
Segmentations and meshes are often prohibitively large to download *en mass* and can have artifacts that generate ambiguities or make attaching annotations like synapses to specific locations unclear.

PCG-skel uses the same chunking that allows the [PyChunkedGraph](https://github.com/seung-lab/PyChunkedGraph) (PCG) to quickly make edits to large, complex neuronal segmentations and, combined with a dynamic caching system that updates after every edit, can generate complete representations of the topology of objects in just a few seconds and minimal data downloads.
Becuase the data is always matched to the underlying representation of the segmentation, there are no ambiguities in what parts are connected to what other or in which vertex a synapse or other annotation is associated with. Light data needs and rapid skeletonization make it useful in environments where analysis is being re-run on frequently changing cells.

However, there is a trade-off in terms of resolution.
The dependency on chunk size means that vertices are roughly a chunk width apart, which in current datasets amounts to about 2 microns.
Thus for understanding the overall structure of a cell or looking at long distance relationships between points along an arbor, these skeletons are quite good, but for detailed analysis at short length scales (0-10 microns or so) where being plus or minus a micron would hurt analysis, we recommend looking at other approaches like kimimaro, meshteasar, or CGAL skeletonization.


## Generating a skeleton

## Generating a meshwork

## Using Features