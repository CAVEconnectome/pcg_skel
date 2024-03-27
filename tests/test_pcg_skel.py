import pytest
import pcg_skel

from conftest import test_client
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
