import datetime
from typing import Optional

import numpy as np
import pandas as pd
from caveclient import CAVEclient
from meshparty import meshwork, skeleton, trimesh_io

from . import features


def rebuild_meshwork(
    root_id: int,
    sk_verts: list,
    sk_edges: list,
    root: int,
    mesh_to_skel_map: list,
    lvl2_ids: list,
    radius: Optional[list] = None,
    compartments: Optional[list] = None,
    metadata: Optional[dict] = None,
    client: Optional[CAVEclient] = None,
    synapses: bool = False,
    timestamp: Optional[datetime.datetime] = None,
    restore_graph: bool = False,
    restore_properties: bool = False,
):
    if metadata is None:
        metadata = {}
    skel = skeleton.Skeleton(
        vertices=sk_verts,
        edges=sk_edges,
        root=root,
        mesh_to_skel_map=mesh_to_skel_map,
        radius=radius,
        meta=metadata,
    )

    l2dict_mesh = {l2id: ii for ii, l2id in enumerate(lvl2_ids)}
    if restore_graph:
        gr = client.chunkedgraph.level2_chunk_graph(root_id)
        mesh_edges = np.apply_along_axis(lambda x: [l2dict_mesh[y] for y in x], 0, gr)

        locs = client.l2cache.get_l2data(lvl2_ids, attributes=["rep_coord_nm"])
        mesh_verts = [locs[str(l2id)].get("rep_coord_nm") for l2id in lvl2_ids]
    else:
        mesh_edges = []
        mesh_verts = sk_verts[mesh_to_skel_map]

    mesh = trimesh_io.Mesh(
        vertices=mesh_verts,
        faces=[[0, 0, 0]],  # Can error without faces,
        link_edges=mesh_edges,
    )

    nrn = meshwork.Meshwork(
        mesh=mesh,
        skeleton=skel,
        seg_id=root_id,
    )

    lvl2_df = pd.DataFrame({"lvl2_id": lvl2_ids, "mesh_ind": np.arange(len(lvl2_ids))})
    nrn.anno.add_annotations("lvl2_ids", lvl2_df, index_column="mesh_ind")

    if compartments is not None:
        compartment_df = pd.DataFrame(
            {"compartment": compartments, "mesh_ind": np.arange(len(compartments))}
        )
        nrn.anno.add_annotations("compartment", compartment_df, index_column="mesh_ind")

    if synapses:
        features.add_synapses(
            nrn,
            synapse_table=client.materialize.synapse_table,
            l2dict_mesh=l2dict_mesh,
            client=client,
            root_id=nrn.seg_id,
            pre=True,
            post=True,
            remove_self_synapse=True,
            timestamp=timestamp,
            live_query=True,
            metadata=False,
        )

    if restore_properties:
        features.add_volumetric_properties(
            nrn,
            client=client,
        )
        features.add_segment_properties(nrn)

    return nrn


def get_meshwork_from_client(
    root_id: int,
    client: CAVEclient,
    synapses: bool = False,
    restore_graph: bool = False,
    restore_properties: bool = False,
    skeleton_version: Optional[int] = 4,
):
    sk = client.skeleton.get_skeleton(
        root_id, skeleton_version=skeleton_version, output_format="dict"
    )
    ts = client.chunkedgraph.get_root_timestamps(root_id, latest=True)[0]
    return rebuild_meshwork(
        root_id=root_id,
        sk_verts=np.array(sk["vertices"]),
        sk_edges=np.array(sk["edges"]),
        root=int(sk["root"]),
        mesh_to_skel_map=np.array(sk["mesh_to_skel_map"]),
        lvl2_ids=np.array(sk["lvl2_ids"]),
        metadata=sk["meta"],
        radius=np.array(sk["radius"]),
        compartments=np.array(sk["compartment"]),
        client=client,
        synapses=synapses,
        timestamp=ts,
        restore_graph=restore_graph,
        restore_properties=restore_properties,
    )
