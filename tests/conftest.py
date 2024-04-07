import pathlib
import pickle

import numpy as np
import pandas as pd
import pytest
from pytest_mock import mocker
from caveclient import CAVEclient

base_path = pathlib.Path(__file__).parent.resolve()

TEST_DATASTACK = "minnie65_public"
MAT_VERSION = 795

INFO_CACHE = {
    "minnie65_public": {
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
    }
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
    client = CAVEclient(TEST_DATASTACK, info_cache=INFO_CACHE)
    mocker.patch.object(client.info, "segmentation_cloudvolume", return_value=test_cv)
    mocker.patch.object(client.l2cache, "get_l2data", return_value=test_l2ids)
    mocker.patch.object(
        client.chunkedgraph, "level2_chunk_graph", return_value=test_l2graph
    )
    synapse_table = mocker.PropertyMock(return_value="synapses_pni_2")
    type(client.materialize).synapse_table = synapse_table
    mocker.patch.object(client.materialize, "get_timestamp", return_value="1234")
    mocker.patch(
        "pcg_skel.pcg_anno.get_level2_synapses",
        return_value=(test_pre_synapses, test_post_synapses),
    )
    return client
