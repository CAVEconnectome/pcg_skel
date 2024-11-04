import pathlib
import pickle

import numpy as np
import pandas as pd
import pytest
from pytest_mock import mocker
from caveclient.tools.testing import CAVEclientMock
from meshparty import meshwork
from packaging.version import Version

base_path = pathlib.Path(__file__).parent.resolve()

TEST_DATASTACK = "minnie65_public"
TEST_DATASTACK_NOCACHE = "minnie65_public_nocache"
MAT_VERSION = 795

INFO_CACHE = {
    TEST_DATASTACK: {
        "proofreading_review_table": None,
        "description": "This is the publicly released version of the minnie65 volume and segmentation. ",
        "proofreading_status_table": None,
        "cell_identification_table": None,
        "local_server": "https://minnie.microns-daf.com",
        "aligned_volume": {
            "description": "This is the second alignment of the IARPA 'minnie65' dataset, completed in the spring of 2020 that used the seamless approach.",
            "name": "minnie65_phase3",
            "id": 1,
            "image_source": "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em",
            "display_name": "Minnie65",
        },
        "synapse_table": "synapses_pni_2",
        "viewer_site": "https://neuroglancer.neuvue.io",
        "analysis_database": None,
        "segmentation_source": "graphene://https://minnie.microns-daf.com/segmentation/table/minnie65_public",
        "soma_table": "nucleus_detection_v0",
        "viewer_resolution_x": 4.0,
        "viewer_resolution_y": 4.0,
        "viewer_resolution_z": 40.0,
    },
    TEST_DATASTACK_NOCACHE: {
        "proofreading_review_table": None,
        "description": "This is the publicly released version of the minnie65 volume and segmentation. ",
        "proofreading_status_table": None,
        "cell_identification_table": None,
        "local_server": "https://minnie.microns-daf.com",
        "aligned_volume": {
            "description": "This is the second alignment of the IARPA 'minnie65' dataset, completed in the spring of 2020 that used the seamless approach.",
            "name": "minnie65_phase3",
            "id": 1,
            "image_source": "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em",
            "display_name": "Minnie65",
        },
        "synapse_table": "synapses_pni_2",
        "viewer_site": "https://neuroglancer.neuvue.io",
        "analysis_database": None,
        "segmentation_source": "graphene://https://minnie.microns-daf.com/segmentation/table/minnie65_public",
        "soma_table": "nucleus_detection_v0",
        "viewer_resolution_x": 4.0,
        "viewer_resolution_y": 4.0,
        "viewer_resolution_z": 40.0,
    },
}


@pytest.fixture
def root_id():
    return 864691136040432126


@pytest.fixture
def center_pt():
    return (177408, 157968, 21002)


class MockCloudVolume:
    @property
    def mip_resolution(self):
        return [4, 4, 40]


test_cv = MockCloudVolume()
with open(base_path / "data/l2_info.pkl", "rb") as f:
    test_l2ids = pickle.load(f)

test_l2graph = np.load(base_path / "data/lvl2_eg.npy")


test_pre_synapses = pd.read_feather(base_path / "data/pre_syn.feather")
test_post_synapses = pd.read_feather(base_path / "data/post_syn.feather")


@pytest.fixture()
def test_l2eg():
    return test_l2graph


@pytest.fixture()
def test_client(mocker):
    client = CAVEclientMock(
        datastack_name=TEST_DATASTACK,
        local_server="https://minnie.microns-daf.com",
        info_file=INFO_CACHE[TEST_DATASTACK],
        chunkedgraph=True,
        materialization=True,
        l2cache=True,
    )

    mocker.patch.object(client.info, "segmentation_cloudvolume", return_value=test_cv)
    mocker.patch.object(client.l2cache, "get_l2data", return_value=test_l2ids)
    mocker.patch.object(client.l2cache, "has_cache", return_value=True)
    mocker.patch.object(
        client.chunkedgraph, "level2_chunk_graph", return_value=test_l2graph
    )
    synapse_table = mocker.PropertyMock(return_value="synapses_pni_2")
    type(client.materialize).synapse_table = synapse_table
    mocker.patch.object(client.materialize, "get_versions", return_value=[1])
    mocker.patch.object(client.materialize, "get_timestamp", return_value="1234")
    mocker.patch(
        "pcg_skel.pcg_anno.get_level2_synapses",
        return_value=(test_pre_synapses, test_post_synapses),
    )
    return client


@pytest.fixture()
def test_client_nol2cache(mocker):
    client = CAVEclientMock(
        datastack_name=TEST_DATASTACK_NOCACHE,
        local_server="https://minnie.microns-daf.com",
        info_file=INFO_CACHE[TEST_DATASTACK_NOCACHE],
        l2cache=True,
        l2cache_disabled=True,
    )
    mocker.patch.object(
        client.materialize, "_get_version", return_value=Version("4.35.0")
    )
    mocker.patch.object(
        client.chunkedgraph, "_get_version", return_value=Version("2.17.2")
    )

    mocker.patch.object(client.info, "segmentation_cloudvolume", return_value=test_cv)
    mocker.patch.object(client.l2cache, "has_cache", return_value=False)
    mocker.patch.object(
        client.chunkedgraph, "level2_chunk_graph", return_value=test_l2graph
    )
    return client


@pytest.fixture()
def test_neuron():
    return meshwork.load_meshwork(base_path / "data" / "test_neuron.h5")
