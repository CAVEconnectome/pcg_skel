import datetime
from typing import Optional

import numpy as np
import pandas as pd
from caveclient import CAVEclient
from meshparty import meshwork, skeleton, trimesh_io

from . import features

compartment_map = {
    "soma": 1,
    "axon": 2,
    "dendrite": 3,
}

def rebuild_meshwork(
    root_id: int,
    sk_verts: np.ndarray,
    sk_edges: np.ndarray,
    root: int,
    mesh_to_skel_map: np.ndarray,
    lvl2_ids: np.ndarray,
    radius: Optional[list] = None,
    compartments: Optional[list] = None,
    is_axon: bool = True,
    metadata: Optional[dict] = None,
    client: Optional[CAVEclient] = None,
    synapses: bool = False,
    timestamp: Optional[datetime.datetime] = None,
    restore_graph: bool = False,
    restore_properties: bool = False,
    synapse_reference_tables: dict = {},
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
            {"compartment": compartments[mesh_to_skel_map], "mesh_ind": np.arange(len(mesh_to_skel_map))}
        )
        nrn.anno.add_annotations("compartment", compartment_df, index_column="mesh_ind")
        if is_axon:
            if compartment_map['axon'] in compartments:
                is_axon = compartment_df.query(f'compartment == {compartment_map["axon"]}')['mesh_ind'].values
                nrn.anno.add_annotations("is_axon", is_axon, mask=True)
            else:
                nrn.anno.add_annotations("is_axon", [], mask=True)
    else:
        if is_axon:
            nrn.anno.add_annotations("is_axon", [], mask=True)

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
            synapse_point_resolution=[1,1,1], # Nanometers, same as skeleton.
            synapse_reference_tables=synapse_reference_tables,
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
    synapse_reference_tables: Optional[dict] = None,
    skeleton_version: Optional[int] = 4,
) -> meshwork.Meshwork:
    """Generate a meshwork file from the information on the skeleton service.
    Note: Requires skeleton service v0.12.3 or higher to be deployed.

    Parameters
    ----------
    root_id : int
        Object root id
    client : CAVEclient
        Initialized caveclient
    synapses : bool, optional
        Whether to add synapses under the annotations "pre_syn" and "post_syn", by default False
    restore_graph : bool, optional
        Whether to restore the level 2 graph, by default False.
        Adds a significant amount of time to the rehydration,
        however the `nrn.mesh` part will not be accurate if this is True.
    restore_properties : bool, optional
        Whether to restore `volume_properties` and `segment_properties`, by default False.
        Adds a significant amount of time to the rehydration.
    synapse_reference_tables : dict, optional
        A dictionary of synapse reference tables to attach to synapses.
        Keys are 'pre' and 'post', values are table names. Defaults to {}.
    skeleton_version : Optional[int], optional
        Which skeleton version to use, by default 4, which is the minimum needed for rehydration to work.

    Returns
    -------
    meshparty.meshwork.MeshWork
        Meshwork object with annotation properties:
        * `compartment`: SWC compartment labels for vertices (1: soma, 2: axon, 3: dendrite)
        * `is_axon`: Vertices with compartment label 2.
        * `lvl2_ids`: Level 2 id for each vertex of the spatial graph.
        * `pre_syn` and `post_syn` (optional): Synapse information.
        * `segment_properties` and `volume_properties` (optional): Segment and volume information.

        Note that the spatial graph (`nrn.mesh`) part of the file will _not_ be correct with this
        function unless `restore_graph` is set to True, but the skeleton and annotations will be.
    """
    sk = client.skeleton.get_skeleton(
        root_id, skeleton_version=skeleton_version, output_format="dict"
    )
    ts = client.chunkedgraph.get_root_timestamps(root_id, latest=True)[0]
    if synapse_reference_tables is None:
        synapse_reference_tables = {}
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
        synapse_reference_tables=synapse_reference_tables,
    )
