from __future__ import annotations
from . import chunk_tools, features
from . import skel_utils as sk_utils

import cloudvolume
import warnings
import numpy as np
import datetime

from caveclient import CAVEclient
from meshparty import meshwork, skeletonize, trimesh_io
from typing import Union, Optional

Numeric = Union[int, float, np.number]

DEFAULT_VOXEL_RESOLUTION = [4, 4, 40]
DEFAULT_COLLAPSE_RADIUS = 7500.0
DEFAULT_INVALIDATION_D = 7500

skeleton_type = "pcg_skel"


def pcg_graph(
    root_id: int,
    client: CAVEclient.frameworkclient.CAVEclientFull,
    cv: cloudvolume.CloudVolume = None,
    return_l2dict: bool = False,
    nan_rounds: int = 10,
    require_complete: bool = False,
    level2_graph: Optional[np.ndarray] = None,
):
    """Compute the level 2 spatial graph (or mesh) of a given root id using the l2cache.

    Some text for you and me.

    Parameters
    ----------
    root_id : int
        Root id of a segment
    client : CAVEclient.caveclient
        Initialized CAVEclient for the dataset.
    cv : cloudvolume.CloudVolume
        Initialized CloudVolume object for the dataset. This does not replace the caveclient, but
        a pre-initizialized cloudvolume can save some time during batch processing.
    return_l2dict : bool
        If True, returns the mappings between l2 ids and vertices.
    nan_rounds : int
        If vertices are missing (or not computed), this sets the number of iterations for smoothing over them.
    require_complete : bool
        If True, raise an Exception if any vertices are missing from the cache.
    level2_graph : np.ndarray, optional
        Level 2 graph for the root id as returned by client.chunkedgraph.level2_chunk_graph.
        A list of lists of edges between level 2 chunks, as defined by their chunk ids.
        If None, will query the chunkedgraph for the level 2 graph. Optional, by default None.


    Returns
    -------
    mesh : meshparty.trimesh_io.Mesh
        Object with a vertex for every level 2 id and edges between all connected level 2 chunks.
    l2dict : dict, optional
        Dictionary with keys as level 2 ids and values as mesh vertex index. Optional, only returned if `return_l2dict` is True.
    l2dict_reverse : dict, optional
        Dictionary with keys as mesh vertex indices and values as level 2 id. Optional, only returned if `return_l2dict` is True.
    """
    if cv is None:
        cv = client.info.segmentation_cloudvolume(progress=False)

    if level2_graph is None:
        lvl2_eg = client.chunkedgraph.level2_chunk_graph(root_id)
    else:
        lvl2_eg = level2_graph

    eg, l2dict_mesh, l2dict_r_mesh, x_ch = chunk_tools.build_spatial_graph(
        lvl2_eg,
        cv,
        client=client,
        method="service",
        require_complete=require_complete,
    )
    mesh_loc = trimesh_io.Mesh(
        vertices=x_ch,
        faces=[[0, 0, 0]],  # Some functions fail if no faces are set.
        link_edges=eg,
    )

    sk_utils.fix_nan_verts_mesh(mesh_loc, nan_rounds)

    if return_l2dict:
        return mesh_loc, l2dict_mesh, l2dict_r_mesh
    else:
        return mesh_loc


def pcg_skeleton_direct(
    vertices,
    edges,
    invalidation_d: Numeric = DEFAULT_INVALIDATION_D,
    root_point: list = None,
    collapse_soma: bool = False,
    collapse_radius: Numeric = DEFAULT_COLLAPSE_RADIUS,
    root_id: Optional[int] = None,
):
    """
    Produce a skeleton from an already-computed l2graph.
    This is effectively a wrapper for meshparty skeletonize with a consistent set of parameters and format.

    Parameters
    ----------
    vertices : np.array
        Array of vertices for the mesh graph.
    edges : np.array
        Array of edges for the mesh graph.
    invalidation_d : int, optional
        Distance (in nanometers) for TEASAR skeleton invalidation.
    root_point : np.array, optional
        3-element list or array with the x,y,z location of the root point in same units as vertices.
        If None, the most distant tip is set to root.
    collapse_soma : bool, optional
        If True, collapse nearby vertices into the root point.
    collapse_radius : int, optional
        Distance (in nanometers) for soma collapse.
    root_id : int, optional
        Root id of the segment, used in metadata. Optional, by default None.
    level2_graph : np.ndarray, optional
        Level 2 graph for the root id as returned by client.chunkedgraph.level2_chunk_graph.
        A list of lists of edges between level 2 chunks, as defined by their chunk ids.
        If None, will query the chunkedgraph for the level 2 graph. Optional, by default None.


    Returns
    -------
    sk : meshparty.skeleton.Skeleton
        Skeleton for the l2graph.
    """

    l2graph = trimesh_io.Mesh(
        vertices=vertices,
        faces=[[0, 0, 0]],  # Some functions fail if no faces are set.
        link_edges=edges,
    )

    sk = skeletonize.skeletonize_mesh(
        l2graph,
        invalidation_d=invalidation_d,
        soma_pt=root_point,
        collapse_soma=collapse_soma,
        soma_radius=collapse_radius,
        compute_radius=False,
        cc_vertex_thresh=0,
        remove_zero_length_edges=True,
        meta={
            "root_id": root_id,
            "skeleton_type": skeleton_type,
            "meta": {"space": "l2cache", "datastack": None},
        },
    )
    return sk


def pcg_skeleton(
    root_id: int,
    client: CAVEclient.frameworkclient.CAVEclientFull,
    datastack_name: str = None,
    cv: cloudvolume.CloudVolume = None,
    invalidation_d: Numeric = 7500,
    return_mesh: bool = False,
    return_l2dict: bool = False,
    return_l2dict_mesh: bool = False,
    root_point: list = None,
    root_point_resolution: list = None,
    collapse_soma: bool = False,
    collapse_radius: Numeric = 7500,
    nan_rounds: int = 10,
    require_complete: bool = False,
    level2_graph: Optional[np.ndarray] = None,
):
    """Produce a skeleton from the level 2 graph.
    Parameters
    ----------
    root_id : int
        Root id of a segment
    client : CAVEclient.caveclient.CAVEclientFull
        Initialized CAVEclient for the dataset.
    datastack_name : string, optional
        If client is None, initializes a CAVEclient at this datastack.
    cv : cloudvolume.CloudVolume, optional
        Initialized CloudVolume object for the dataset. This does not replace the caveclient, but
        a pre-initizialized cloudvolume can save some time during batch processing.
    invalidation_d : int, optional
        Distance (in nanometers) for TEASAR skeleton invalidation.
    return_mesh : bool, optional
        If True, returns the mesh graph as well as the skeleton.
    return_l2dict : bool, optional
        If True, returns the mappings between l2 ids and skeleton vertices.
    return_l2dict_mesh : bool, optional
        If True, returns mappings between l2 ids and mesh graph vertices.
    root_point : npt.ArrayLike, optional
        3-element list or array with the x,y,z location of the root point.
        If None, the most distant tip is set to root.
    root_point_resolution : npt.ArrayLike, optional
        3-element list or array with the x,y,z resolution of the root point, in nanometers per voxel dimension.
    collapse_soma : bool, optional
        If True, collapse nearby vertices into the root point.
    collapse_radius : int, optional
        Distance (in nanometers) for soma collapse.
    nan_rounds : int, optional
        If vertices are missing (or not computed), this sets the number of iterations for smoothing over them.
    require_complete : bool, optional
        If True, raise an Exception if any vertices are missing from the cache.
    level2_graph : np.ndarray, optional
        Level 2 graph for the root id as returned by client.chunkedgraph.level2_chunk_graph.
        A list of lists of edges between level 2 chunks, as defined by their chunk ids.
        If None, will query the chunkedgraph for the level 2 graph. Optional, by default None.

    Returns
    -------
    sk : meshparty.skeleton.Skeleton
        Skeleton for the root id
    mesh : meshparty.trimesh_io.Mesh, optional
        Mesh graph that the skeleton is based on, only returned if return_mesh is True.
    (l2dict_skel, l2dict_reverse): (dict, dict), optional
        Dictionaries mapping l2 ids to skeleton vertices and skeleton vertices to l2 ids, respectively. Only returned if return_l2dict is True.
    (l2dict_mesh, l2dict_mesh): (dict, dict), optional
        Dictionaries mapping l2 ids to mesh graph vertices and mesh_graph vertices to l2 ids, respectively. Only returned if return_l2dict is True.
    """
    if client is None:
        client = CAVEclient(datastack_name)
    if cv is None:
        cv = client.info.segmentation_cloudvolume(progress=False)

    if root_point_resolution is None:
        root_point_resolution = cv.mip_resolution(0)
    if root_point is not None:
        root_point = np.array(root_point) * root_point_resolution

    mesh, l2dict_mesh, l2dict_r_mesh = pcg_graph(
        root_id,
        client=client,
        return_l2dict=True,
        nan_rounds=nan_rounds,
        require_complete=require_complete,
        level2_graph=level2_graph,
    )

    metameta = {"space": "l2cache", "datastack": client.datastack_name}

    sk = skeletonize.skeletonize_mesh(
        mesh,
        invalidation_d=invalidation_d,
        soma_pt=root_point,
        collapse_soma=collapse_soma,
        soma_radius=collapse_radius,
        compute_radius=False,
        cc_vertex_thresh=0,
        remove_zero_length_edges=True,
        meta={
            "root_id": root_id,
            "skeleton_type": skeleton_type,
            "meta": metameta,
        },
    )

    l2dict, l2dict_r = sk_utils.filter_l2dict(sk, l2dict_r_mesh)

    out_list = [sk]
    if return_mesh:
        out_list.append(mesh)
    if return_l2dict:
        out_list.append((l2dict, l2dict_r))
    if return_l2dict_mesh:
        out_list.append((l2dict_mesh, l2dict_r_mesh))
    if len(out_list) == 1:
        return out_list[0]
    else:
        return tuple(out_list)


def pcg_meshwork(
    root_id: int,
    datastack_name: Optional[str] = None,
    client: Optional[CAVEclient] = None,
    cv: Optional[cloudvolume.CloudVolume] = None,
    root_point: Optional[list] = None,
    root_point_resolution: Optional[list] = None,
    collapse_soma: bool = False,
    collapse_radius: Numeric = DEFAULT_COLLAPSE_RADIUS,
    synapses: Optional[Union[bool, str]] = None,
    synapse_table: Optional[str] = None,
    remove_self_synapse: bool = True,
    live_query: bool = False,
    timestamp: Optional[datetime.datetime] = None,
    invalidation_d: Numeric = DEFAULT_INVALIDATION_D,
    require_complete: bool = False,
    metadata: bool = False,
    synapse_partners: bool = False,
    synapse_point_resolution: list = [1, 1, 1],
    synapse_representative_point_pre: str = "ctr_pt_position",
    synapse_representative_point_post: str = "ctr_pt_position",
    level2_graph: Optional[np.ndarray] = None,
) -> meshwork.Meshwork:
    """Generate a meshwork file based on the level 2 graph.

    Parameters
    ----------
    root_id : int
        Root id of an object in the pychunkedgraph.
    datastack_name : str or None, optional
        Datastack name to use to initialize a client, if none is provided. By default None.
    client : caveclient.CAVEclientFull or None, optional
        Initialized CAVE client. If None is given, will use the datastack_name to create one. By default None
    cv : cloudvolume.CloudVolume or None, optional
        Initialized cloudvolume. If none is given, the client info will be used to create one. By default None
    root_point : array-like or None, optional
        3 element xyz location for the location to set the root in units set by root_point_resolution,
        by default None. If None, a distal tip is selected.
    root_point_resolution : array-like, optional
        Resolution in euclidean space of the root_point, by default [4, 4, 40]
    collapse_soma : bool, optional,
        If True, collapses vertices within a given radius of the root point into the root vertex, typically to better
        represent primary neurite branches. Requires a specified root_point. Default if False.
    collapse_radius : float, optional
        Max distance in euclidean space for soma collapse. Default is 10,000 nm (10 microns).
    synapses : 'pre', 'post', 'all', True, or None, optional
        If not None, queries the synapse_table for presynaptic synapses (if 'pre'),  postsynaptic sites (if 'post'), or both (if 'all' or True). By default None
    synapse_table : str, optional
        Name of the synapse table to query if synapses are requested, by default None
    remove_self_synapse : bool, optional
        If True, filters out synapses whose pre- and postsynaptic root ids are the same neuron, by default True
    live_query : bool, optional
        If True, expect a timestamp for querying at a give point in time. Otherwise, use the materializatio set by the client. Optional, by default False.
    timestamp = datetime.datetime, optional
        If set, acts as the time at which all root ids and annotations are found at.
    invalidation_d : int, optional
        Invalidation radius in hops for the mesh skeletonization along the chunk adjacency graph, by default 3
    require_complete : bool, optional
        If True, raise an Exception if any vertices are missing from the cache, by default False
    metadata : bool, optional
        If True, adds metadata to the meshwork annotations. By default False.
    synapse_partners : bool, optional
        If True, includes the partner root id to the synapse annotation. By default False, because partner roots can change across time.
    synapse_point_resolution : array-like, optional
        Resolution in euclidean space of the synapse points, by default None. If None, the resolution will be the default of the synapse table.
    synapse_representative_point_pre : str, optional
        If set, uses the specified column in the synapse table for the pre-synaptic points. By default 'ctr_pt_position'.
    synapse_representative_point_post : str, optional
        If set, uses the specified column in the synapse table for the post-synaptic points. By default 'ctr_pt_position'.

    Returns
    -------
    meshparty.meshwork.Meshwork
        Meshwork object with skeleton based on the level 2 graph. See documentation for details.
    """
    if client is None:
        client = CAVEclient(datastack_name)
    if cv is None:
        cv = client.info.segmentation_cloudvolume(progress=True, parallel=1)
    if root_point_resolution is None:
        root_point_resolution = cv.mip_resolution(0)
    if synapse_table is None:
        synapse_table = client.materialize.synapse_table

    sk, mesh, (l2dict_mesh, l2dict_mesh_r) = pcg_skeleton(
        root_id,
        client=client,
        cv=cv,
        root_point=root_point,
        root_point_resolution=root_point_resolution,
        collapse_soma=collapse_soma,
        collapse_radius=collapse_radius,
        invalidation_d=invalidation_d,
        return_mesh=True,
        return_l2dict_mesh=True,
        require_complete=require_complete,
        level2_graph=level2_graph,
    )

    nrn = meshwork.Meshwork(mesh, seg_id=root_id, skeleton=sk)

    pre, post = False, False
    if synapses is not None and synapse_table is not None:
        if synapses == "pre":
            pre, post = True, False
        elif synapses == "post":
            pre, post = False, True
        elif synapses == "all" or synapses is True:
            pre, post = True, True
        else:
            raise ValueError('Synapses must be one of "pre", "post", or "all" or True.')

        if not timestamp:
            timestamp = client.materialize.get_timestamp()

        features.add_synapses(
            nrn,
            synapse_table,
            l2dict_mesh,
            client,
            root_id=root_id,
            pre=pre,
            post=post,
            remove_self_synapse=remove_self_synapse,
            timestamp=timestamp,
            live_query=live_query,
            metadata=metadata,
            synapse_partners=synapse_partners,
            synapse_point_resolution=synapse_point_resolution,
            synapse_representative_point_pre=synapse_representative_point_pre,
            synapse_representative_point_post=synapse_representative_point_post,
        )

    features.add_lvl2_ids(nrn, l2dict_mesh)
    return nrn


def coord_space_skeleton(
    root_id,
    client,
    datastack_name=None,
    cv=None,
    invalidation_d=10_000,
    return_mesh=False,
    return_l2dict=False,
    return_l2dict_mesh=False,
    root_point=None,
    root_point_resolution=None,
    collapse_soma=False,
    collapse_radius=7500,
    nan_rounds=10,
    require_complete=False,
):
    """Produce a skeleton from the level 2 graph.

    **Deprecated: Please use pcg_skeleton instead.**

    Parameters
    ----------
    root_id : int
        Root id of a segment
    client : CAVEclient.caveclient
        Initialized CAVEclient for the dataset.
    datastack_name : string, optional
        If client is None, initializes a CAVEclient at this datastack, by default None.
    cv : cloudvolume.CloudVolume, optional
        Initialized CloudVolume object for the dataset. This does not replace the caveclient, but
        a pre-initizialized cloudvolume can save some time during batch processing. By default None.
    invalidation_d : int, optional
        Distance (in nanometers) for TEASAR skeleton invalidation, by default 10_000.
    return_mesh : bool, optional
        If True, returns the mesh graph as well as the skeleton, by default False
    return_l2dict : bool, optional
        If True, returns the mappings between l2 ids and skeleton vertices, by default False
    return_l2dict_mesh : bool, optional
        If True, returns mappings between l2 ids and mesh graph vertices, by default False
    root_point : list-like, optional
        3-element list or array with the x,y,z location of the root point. Optional, by default None.
        If None, the most distant tip is set to root.
    root_point_resolution : list-like, optional
        3-element list or array with the x,y,z resolution of the root point, in nanometers per voxel dimension, by default None.
    collapse_soma : bool, optional
        If True, collapse nearby vertices into the root point, by default False.
    collapse_radius : int, optional
        Distance (in nanometers) for soma collapse, by default 7500.
    nan_rounds : int, optional
        If vertices are missing (or not computed), this sets the number of iterations for smoothing over them. By default 10
    require_complete : bool, optional
        If True, raise an Exception if any vertices are missing from the cache, by default False

    Returns
    -------
    sk : meshparty.skeleton.Skeleton
        Skeleton for the root id
    mesh : meshparty.trimesh_io.Mesh (optional)
        Mesh graph that the skeleton is based on, only returned if return_mesh is True.
    (l2dict_skel, l2dict_reverse): tuple of dicts (optional)
        Dictionaries mapping l2 ids to skeleton vertices and skeleton vertices to l2 ids, respectively. Only returned if return_l2dict is True.
    (l2dict_mesh, l2dict_mesh): tuple of dicts (optional)
        Dictionaries mapping l2 ids to mesh graph vertices and mesh_graph vertices to l2 ids, respectively. Only returned if return_l2dict is True.
    """
    warnings.warn(
        "The function `coord_space_skeleton` is deprecated and has been replaced with 'pcg_skeleton'",
        DeprecationWarning,
    )
    return pcg_skeleton(
        root_id,
        client,
        datastack_name=datastack_name,
        cv=cv,
        invalidation_d=invalidation_d,
        return_mesh=return_mesh,
        return_l2dict=return_l2dict,
        return_l2dict_mesh=return_l2dict_mesh,
        root_point=root_point,
        root_point_resolution=root_point_resolution,
        collapse_soma=collapse_soma,
        collapse_radius=collapse_radius,
        nan_rounds=nan_rounds,
        require_complete=require_complete,
    )


def coord_space_mesh(
    root_id,
    client,
    cv=None,
    return_l2dict=False,
    nan_rounds=10,
    require_complete=False,
):
    """Compute the level 2 spatial graph (or mesh) of a given root id using the l2cache.
    Deprecated: Use pcg_graph instead.

    Parameters
    ----------
    root_id : int
        Root id of a segment
    client : CAVEclient.caveclient
        Initialized CAVEclient for the dataset.
    cv : cloudvolume.CloudVolume, optional
        Initialized CloudVolume object for the dataset. This does not replace the caveclient, but
        a pre-initizialized cloudvolume can save some time during batch processing. By default None.
    return_l2dict : bool, optional
        If True, returns the mappings between l2 ids and vertices, by default False
    nan_rounds : int, optional
        If vertices are missing (or not computed), this sets the number of iterations for smoothing over them. By default 10
    require_complete : bool, optional
        If True, raise an Exception if any vertices are missing from the cache, by default False

    Returns
    -------
    mesh : meshparty.trimesh_io.Mesh
        Object with a vertex for every level 2 id and edges between all connected level 2 chunks.
    l2dict : dict (optional)
        Dictionary with keys as level 2 ids and values as mesh vertex index. Optional, only returned if `return_l2dict` is True.
    l2dict_reverse : dict (optional)
        Dictionary with keys as mesh vertex indices and values as level 2 id. Optional, only returned if `return_l2dict` is True.
    """

    warnings.warn(
        "The function `coord_space_mesh` is deprecated and has been replaced with 'pcg_graph'",
        DeprecationWarning,
    )
    return pcg_graph(
        root_id,
        client,
        cv=cv,
        return_l2dict=return_l2dict,
        nan_rounds=nan_rounds,
        require_complete=require_complete,
    )


def coord_space_meshwork(
    root_id,
    datastack_name=None,
    client=None,
    cv=None,
    root_point=None,
    root_point_resolution=None,
    collapse_soma=False,
    collapse_radius=DEFAULT_COLLAPSE_RADIUS,
    synapses=None,
    synapse_table=None,
    remove_self_synapse=True,
    live_query=False,
    timestamp=None,
    invalidation_d=DEFAULT_INVALIDATION_D,
    require_complete=False,
    metadata=False,
):
    """Generate a meshwork file based on the level 2 graph.

    Parameters
    ----------
    root_id : int
        Root id of an object in the pychunkedgraph.
    datastack_name : str or None, optional
        Datastack name to use to initialize a client, if none is provided. By default None.
    client : caveclient.CAVEclientFull or None, optional
        Initialized CAVE client. If None is given, will use the datastack_name to create one. By default None
    cv : cloudvolume.CloudVolume or None, optional
        Initialized cloudvolume. If none is given, the client info will be used to create one. By default None
    root_point : array-like or None, optional
        3 element xyz location for the location to set the root in units set by root_point_resolution,
        by default None. If None, a distal tip is selected.
    root_point_resolution : array-like, optional
        Resolution in euclidean space of the root_point, by default [4, 4, 40]
    collapse_soma : bool, optional,
        If True, collapses vertices within a given radius of the root point into the root vertex, typically to better
        represent primary neurite branches. Requires a specified root_point. Default if False.
    collapse_radius : float, optional
        Max distance in euclidean space for soma collapse. Default is 10,000 nm (10 microns).
    synapses : 'pre', 'post', 'all', or None, optional
        If not None, queries the synapse_table for presynaptic synapses (if 'pre'),  postsynaptic sites (if 'post'), or both (if 'all'). By default None
    synapse_table : str, optional
        Name of the synapse table to query if synapses are requested, by default None
    remove_self_synapse : bool, optional
        If True, filters out synapses whose pre- and postsynaptic root ids are the same neuron, by default True
    live_query : bool, optional
        If True, expect a timestamp for querying at a give point in time. Otherwise, use the materializatio set by the client. Optional, by default False.
    timestamp = datetime.datetime, optional
        If set, acts as the time at which all root ids and annotations are found at.
    invalidation_d : int, optional
        Invalidation radius in hops for the mesh skeletonization along the chunk adjacency graph, by default 3
    require_complete : bool, optional
        If True, raise an Exception if any vertices are missing from the cache, by default False
    level2_graph : np.ndarray, optional
        Level 2 graph for the root id as returned by client.chunkedgraph.level2_chunk_graph.
        A list of lists of edges between level 2 chunks, as defined by their chunk ids.
        If None, will query the chunkedgraph for the level 2 graph. Optional, by default None.


    Returns
    -------
    meshparty.meshwork.Meshwork
        Meshwork object with skeleton based on the level 2 graph. See documentation for details.
    """
    warnings.warn(
        "The function `coord_space_meshwork` is deprecated and has been replaced with 'pcg_meshwork'",
        DeprecationWarning,
    )

    return pcg_meshwork(
        root_id,
        datastack_name=datastack_name,
        client=client,
        cv=cv,
        root_point=root_point,
        root_point_resolution=root_point_resolution,
        collapse_soma=collapse_soma,
        collapse_radius=collapse_radius,
        synapses=synapses,
        synapse_table=synapse_table,
        remove_self_synapse=remove_self_synapse,
        live_query=live_query,
        timestamp=timestamp,
        invalidation_d=invalidation_d,
        require_complete=require_complete,
        metadata=metadata,
    )
