import pytest
import pcg_skel
from pcg_skel.chunk_cache import NoL2CacheException
import numpy as np

from conftest import test_client, test_client_nol2cache
from conftest import root_id, center_pt


def test_pcg_graph(test_client, root_id):
    graph_m = pcg_skel.pcg_graph(root_id, test_client)
    assert graph_m.vertices is not None
    assert graph_m.edges is not None


def test_pcg_skeleton(test_client, root_id, center_pt):
    graph_sk = pcg_skel.pcg_skeleton(
        root_id,
        test_client,
        collapse_radius=True,
        root_point=center_pt,
        root_point_resolution=[4, 4, 40],
    )
    assert graph_sk.vertices is not None
    assert graph_sk.edges is not None
    assert graph_sk.path_length() > 0


def test_pcg_skeleton_prebaked(test_client, root_id, center_pt, test_l2eg):
    graph_sk = pcg_skel.pcg_skeleton(
        root_id,
        test_client,
        collapse_radius=True,
        root_point=center_pt,
        root_point_resolution=[4, 4, 40],
        level2_graph=test_l2eg,
    )
    assert graph_sk.vertices is not None
    assert graph_sk.edges is not None
    assert graph_sk.path_length() > 0


def test_pcg_skeleton_direct(test_client, root_id, center_pt):
    graph_m = pcg_skel.pcg_graph(root_id, test_client)
    sk = pcg_skel.pcg_skeleton_direct(
        vertices=graph_m.vertices,
        edges=graph_m.graph_edges,
    )
    assert len(sk.vertices) == 2567


def test_pcg_meshwork(test_client, root_id, center_pt):
    nrn = pcg_skel.pcg_meshwork(
        root_id=root_id,
        client=test_client,
        collapse_soma=True,
        root_point=center_pt,
        root_point_resolution=[4, 4, 40],
        synapses="all",
        synapse_table=test_client.materialize.synapse_table,
    )
    assert len(nrn.anno.table_names) > 0
    assert nrn.mesh.vertices.shape[0] > 0


def test_defunct_meshwork(test_client, root_id, center_pt):
    nrn2 = pcg_skel.coord_space_meshwork(
        root_id=root_id,
        client=test_client,
        collapse_soma=True,
        root_point=center_pt,
        root_point_resolution=[4, 4, 40],
        synapses="all",
        synapse_table=test_client.materialize.synapse_table,
    )
    assert len(nrn2.anno.table_names) > 0
    assert nrn2.mesh.vertices.shape[0] > 0


def test_pcg_meshwork_noclient(test_client_nol2cache, root_id, center_pt):
    with pytest.raises(NoL2CacheException):
        pcg_skel.pcg_skeleton(
            root_id,
            test_client_nol2cache,
            collapse_radius=True,
            root_point=center_pt,
            root_point_resolution=[4, 4, 40],
        )


def test_feature_aggregation(test_neuron):
    orig_df = test_neuron.anno["synapse_count"].df
    skel_df = pcg_skel.features.aggregate_property_to_skeleton(
        test_neuron,
        "synapse_count",
        agg_dict={
            "num_syn_out": "sum",
            "num_syn_in": "sum",
            "net_size_out": "sum",
            "net_size_in": "sum",
        },
    )
    for col in ["num_syn_in", "num_syn_out", "net_size_in", "net_size_out"]:
        assert skel_df.sum()[col] == orig_df.sum()[col]
