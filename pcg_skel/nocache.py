import numpy as np
from caveclient import CAVEclient
from meshparty import skeleton, skeletonize, trimesh_io, meshwork

from . import chunk_tools, features
from . import skel_utils as sk_utils
from . import utils

DEFAULT_VOXEL_RESOLUTION = [4, 4, 40]
DEFAULT_COLLAPSE_RADIUS = 7500.0
DEFAULT_INVALIDATION_D = 7500

skeleton_type = "pcg_skel"


def chunk_index_mesh(
    root_id,
    client=None,
    datastack_name=None,
    cv=None,
    return_l2dict=False,
):
    """Download a mesh with chunk index vertices

    Parameters
    ----------
    root_id : int
        Root id to download.
    client : CAVEclient, optional
        Preset CAVEclient, by default None.
    datastack_name : str or None, optional
        Datastack to use to initialize a CAVEclient, by default None.
    cv : cloudvolume.CloudVolume or None, optional
        Cloudvolume instance, by default None.
    return_l2dict : bool, optional
        If True, returns both a l2id to vertex dict and the reverse, by default False.

    Returns
    -------
    mesh : trimesh_io.Mesh
        Chunk graph represented as a mesh, with vertices at chunk index locations and edges in the link_edges attribute.
    l2dict_mesh : dict
        l2 id to mesh vertex index dictionary. Only returned if return_l2dict is True.
    l2dict_r_mesh : dict
        Mesh vertex index to l2 id dictionary. Only returned if return_l2dict is True.
    """

    if client is None:
        client = CAVEclient(datastack_name)
    if cv is None:
        cv = client.info.segmentation_cloudvolume(progress=False)

    lvl2_eg = client.chunkedgraph.level2_chunk_graph(root_id)
    eg, l2dict_mesh, l2dict_r_mesh, x_ch = chunk_tools.build_spatial_graph(lvl2_eg, cv)
    mesh_chunk = trimesh_io.Mesh(
        vertices=x_ch,
        faces=[[0, 0, 0]],  # Some functions fail if no faces are set.
        link_edges=eg,
    )
    if return_l2dict:
        return mesh_chunk, l2dict_mesh, l2dict_r_mesh
    else:
        return mesh_chunk


def chunk_index_skeleton(
    root_id,
    client=None,
    datastack_name=None,
    cv=None,
    root_point=None,
    invalidation_d=3,
    return_mesh=False,
    return_l2dict=False,
    return_mesh_l2dict=False,
    root_point_resolution=None,
    root_point_search_radius=300,
    n_parallel=1,
):
    """Generate a basic skeleton with chunked-graph index vertices.

    Parameters
    ----------
    root_id : np.uint64
        Neuron root id
    client : caveclient.CAVEclient, optional
        CAVEclient for a datastack, by default None. If None, you must specify a datastack name.
    datastack_name : str, optional
        Datastack name to create a CAVEclient, by default None. Only used if client is None.
    cv : cloudvolume.CloudVolume, optional
        CloudVolume associated with the object, by default None. If None, one is created based on the client info.
    root_point : array, optional
        Point in voxel space to set the root vertex. By default None, which makes a random tip root.
    invalidation_d : int, optional
        TEASAR invalidation radius in chunk space, by default 3
    return_mesh : bool, optional
        If True, returns the pre-skeletonization mesh with vertices in chunk index space, by default False
    return_l2dict : bool, optional
        If True, returns the level 2 id to vertex index dict. By default True
    n_parallel : int, optional
        Sets number of parallel threads for cloudvolume, by default 1

    Returns
    -------
    sk : meshparty.skeleton.Skeleton
        Skeleton object
    mesh : meshparty.trimesh_io.Mesh
        Mesh object, only if return_mesh is True
    level2_dict : dict
        Level 2 id to vertex map, only if return_l2dict is True.
    """

    if client is None:
        client = CAVEclient(datastack_name)
    if n_parallel is None:
        n_parallel = 1
    if cv is None:
        cv = client.info.segmentation_cloudvolume(progress=True, parallel=1)

    if root_point_resolution is None:
        root_point_resolution = cv.mip_resolution(0)

    mesh_chunk, l2dict_mesh, l2dict_r_mesh = chunk_index_mesh(
        root_id, client=client, cv=cv, return_l2dict=True
    )

    if root_point is not None:
        lvl2_root_chid, lvl2_root_loc = chunk_tools.get_closest_lvl2_chunk(
            root_point,
            root_id,
            client=client,
            cv=None,
            radius=root_point_search_radius,
            voxel_resolution=root_point_resolution,
            return_point=True,
        )  # Need to have cv=None because of a cloudvolume inconsistency
        root_mesh_index = l2dict_mesh[lvl2_root_chid]
    else:
        root_mesh_index = None

    metameta = {"space": "chunk", "datastack": client.datastack_name}
    sk_ch = skeletonize.skeletonize_mesh(
        mesh_chunk,
        invalidation_d=invalidation_d,
        collapse_soma=False,
        compute_radius=False,
        cc_vertex_thresh=0,
        root_index=root_mesh_index,
        remove_zero_length_edges=False,
        meta={
            "root_id": root_id,
            "skeleton_type": skeleton_type,
            "meta": metameta,
        },
    )

    l2dict, l2dict_r = sk_utils.filter_l2dict(sk_ch, l2dict_r_mesh)

    out_list = [sk_ch]
    if return_mesh:
        out_list.append(mesh_chunk)
    if return_l2dict:
        out_list.append((l2dict, l2dict_r))
    if return_mesh_l2dict:
        out_list.append((l2dict_mesh, l2dict_r_mesh))
    if len(out_list) == 1:
        return out_list[0]
    else:
        return tuple(out_list)


def refine_chunk_index_skeleton(
    sk_ch,
    l2dict_reversed,
    cv,
    refine_inds="all",
    scale_chunk_index=True,
    root_location=None,
    nan_rounds=20,
    return_missing_ids=False,
    segmentation_fallback=False,
    fallback_mip=2,
    cache=None,
    save_to_cache=False,
    client=None,
    l2cache=False,
):
    """Refine skeletons in chunk index space to Euclidean space.

    Parameters
    ----------
    sk_ch : meshparty.skeleton.Skeleton
        Skeleton in chunk index space
    l2dict_reversed : dict
        Mapping between skeleton vertex index and level 2 id.
    cv : cloudvolume.CloudVolume
        Associated cloudvolume
    refine_inds : str, None or list-like, optional
        Skeleton indices to refine, 'all', or None. If 'all', does all skeleton indices.
        If None, downloads no index but can use other options.
        By default 'all'.
    scale_chunk_index : bool, optional
        If True, maps unrefined chunk index locations to the center of the chunk in
        Euclidean space, by default True
    root_location : list-like, optional
        3-element euclidean space location to which to map the root vertex location, by default None
    nan_rounds : int, optional
        Number of passes to smooth over any missing values by averaging proximate vertex locations.
        Only used if refine_inds is 'all'. Default is 20.
    return_missing_ids : bool, optional
        If True, returns ids of any missing level 2 meshes. Default is False
    segmentation_fallback : bool, optional
        If True, downloads the segmentation at mip level in fallback_mip to get a location. Very slow. Default is False.
    fallback_mip : int, optional
        The mip level used in segmentation fallback. Default is 2.
    cache : str, optional
        If set to 'service', uses the l2cache service if available available. Otherwise, a filename for a sqlite database storing locations associated with level 2 ids. Default is None.
    save_to_cache : bool, optional
        If using a sqlite database, setting this to True will add values to the cache as downloads occur.
    client : CAVEclient, optional
        If using the l2cache service, provides a client that can access it.
    l2cache : bool, optional,
        Set to True if using a l2cache to localize vertices. Same as setting cache to 'service'. Default is False.

    Returns
    -------
    meshparty.skeleton.Skeleton
        Skeleton with remapped vertex locations
    """
    if nan_rounds is None:
        convert_missing = True
    else:
        convert_missing = False

    if l2cache:
        cache = "service"

    refine_out = chunk_tools.refine_vertices(
        sk_ch.vertices,
        l2dict_reversed=l2dict_reversed,
        cv=cv,
        refine_inds=refine_inds,
        scale_chunk_index=scale_chunk_index,
        convert_missing=convert_missing,
        return_missing_ids=return_missing_ids,
        segmentation_fallback=segmentation_fallback,
        fallback_mip=fallback_mip,
        cache=cache,
        save_to_cache=save_to_cache,
        client=client,
    )
    if return_missing_ids:
        new_verts, missing_ids = refine_out
    else:
        new_verts = refine_out

    if root_location is not None:
        new_verts[sk_ch.root] = root_location

    l2_sk = skeleton.Skeleton(
        vertices=new_verts,
        edges=sk_ch.edges,
        root=sk_ch.root,
        remove_zero_length_edges=False,
        mesh_index=sk_ch.mesh_index,
        mesh_to_skel_map=sk_ch.mesh_to_skel_map,
        meta=sk_ch.meta,
    )
    metameta = {
        "space": "euclidean",
    }
    try:
        l2_sk.meta.update_metameta(metameta)
    except:
        pass

    if isinstance(refine_inds, str) and refine_inds == "all":
        sk_utils.fix_nan_verts(l2_sk, num_rounds=nan_rounds)

    if return_missing_ids:
        return l2_sk, missing_ids
    else:
        return l2_sk


def pcg_skeleton(
    root_id,
    client=None,
    datastack_name=None,
    cv=None,
    refine="all",
    root_point=None,
    root_point_resolution=None,
    root_point_search_radius=300,
    collapse_soma=False,
    collapse_radius=10_000.0,
    invalidation_d=3,
    return_mesh=False,
    return_l2dict=False,
    return_l2dict_mesh=False,
    return_missing_ids=False,
    nan_rounds=20,
    segmentation_fallback=False,
    fallback_mip=2,
    cache=None,
    save_to_cache=False,
    n_parallel=1,
    l2cache=False,
):
    """Create a euclidean-space skeleton from the pychunkedgraph

    Parameters
    ----------
    root_id : uint64
        Root id of the neuron to skeletonize
    client : caveclient.CAVEclientFull or None, optional
        Pre-specified cave client for the pcg. If this is not set, datastack_name must be provided. By default None
    datastack_name : str or None, optional
        If no client is specified, a CAVEclient is created with this datastack name, by default None
    cv : cloudvolume.CloudVolume or None, optional
        Prespecified cloudvolume instance. If None, uses the client info to make one, by default None
    refine : 'all', 'ep', 'bp', 'epbp', 'bpep', or None, optional
        Selects how to refine vertex locations by downloading mesh chunks. Unrefined vertices are placed in the
        center of their chunk in euclidean space.
        * 'all' refines all vertex locations. (Default)
        * 'ep' refines end points only
        * 'bp' refines branch points only
        * 'bpep' or 'epbp' refines both branch and end points.
        * None refines no points.
        * 'chunk' Keeps things in chunk index space.
    root_point : array-like or None, optional
        3 element xyz location for the location to set the root in units set by root_point_resolution,
        by default None. If None, a distal tip is selected.
    root_point_resolution : array-like, optional
        Resolution in euclidean space of the root_point, by default [4, 4, 40]
    root_point_search_radius : int, optional
        Distance in euclidean space to look for segmentation when finding the root vertex, by default 300
    collapse_soma : bool, optional,
        If True, collapses vertices within a given radius of the root point into the root vertex, typically to better
        represent primary neurite branches. Requires a specified root_point. Default if False.
    collapse_radius : float, optional
        Max distance in euclidean space for soma collapse. Default is 10,000 nm (10 microns).
    invalidation_d : int, optional
        Invalidation radius in hops for the mesh skeletonization along the chunk adjacency graph, by default 3
    return_mesh : bool, optional
        If True, returns the mesh in chunk index space, by default False
    return_l2dict : bool, optional
        If True, returns the tuple (l2dict, l2dict_r), by default False.
        l2dict maps all neuron level2 ids to skeleton vertices. l2dict_r maps skeleton indices to their direct level 2 id.
    return_l2dict_mesh : bool, optional
        If True, returns the tuple (l2dict_mesh, l2dict_mesh_r), by default False.
        l2dict_mesh maps neuron level 2 ids to mesh vertices, l2dict_r maps mesh indices to level 2 ids.
    return_missing_ids : bool, optional
        If True, returns level 2 ids that were missing in the chunkedgraph, by default False. This can be useful
        for submitting remesh requests in case of errors.
    nan_rounds : int, optional
        Maximum number of rounds of smoothing to eliminate missing vertex locations in the event of a
        missing level 2 mesh, by default 20. This is only used when refine=='all'.
    segmentation_fallback : bool, optional
        If True, uses the segmentation in cases of missing level 2 meshes. This is slower but more robust.
        Default is True.
    cache : str or None, optional
        If set to 'service', uses the online l2cache service (if available). Otherwise, this is the filename of a sqlite database with cached lookups for l2 ids. Optional, default is None.
    n_parallel : int, optional
        Number of parallel downloads passed to cloudvolume, by default 1
    l2cache : bool, optional
        Set to True if using the l2cache. Equivalent to cache='service'. Default is False.

    Returns
    -------
    sk_l2 : meshparty.skeleton.Skeleton
        Skeleton with vertices in euclidean space
    mesh_l2 : meshparty.mesh.Mesh, optional
        Mesh with vertices in chunk index space. Only if return_mesh is True.
    (l2dict, l2dict_r) : (dict, dict), optional
        Mappings between level 2 ids and skeleton indices. Only if return_l2dict is True.
    (l2dict_mesh, l2dict_mesh_r) : (dict, dict), optional
        Mappings between level 2 ids and mesh indices. Only if return_l2dict_mesh is True.
    missing_ids : np.array, optional
        List of level 2 ids with missing mesh fragments. Only if return_missing_ids is True.
    """

    if client is None:
        client = CAVEclient(datastack_name)
    if n_parallel is None:
        n_parallel = 1
    if cv is None:
        cv = client.info.segmentation_cloudvolume(progress=True, parallel=1)

    if root_point_resolution is None:
        root_point_resolution = cv.mip_resolution(0)

    (
        sk_ch,
        mesh_ch,
        (l2dict, l2dict_r),
        (l2dict_mesh, l2dict_mesh_r),
    ) = chunk_index_skeleton(
        root_id,
        client=client,
        datastack_name=datastack_name,
        cv=cv,
        root_point=root_point,
        root_point_resolution=root_point_resolution,
        root_point_search_radius=root_point_search_radius,
        invalidation_d=invalidation_d,
        return_mesh=True,
        return_mesh_l2dict=True,
        return_l2dict=True,
        n_parallel=n_parallel,
    )
    if refine == "all":
        refine_inds = "all"
    elif refine == "bp":
        refine_inds = sk_ch.branch_points_undirected
    elif refine == "ep":
        refine_inds = sk_ch.end_points_undirected
    elif refine == "epbp" or refine == "bpep":
        refine_inds = np.concatenate(
            (sk_ch.end_points_undirected, sk_ch.branch_points_undirected)
        )
    elif refine == "chunk":
        refine_inds = None
    elif refine is None:
        refine_inds = None
    else:
        raise ValueError(
            '"refine" must be one of "all", "bp", "ep", "epbp"/"bpep", "chunk", or None'
        )

    if root_point is not None:
        root_point_euc = root_point * np.array([root_point_resolution])
    else:
        root_point_euc = None

    sk_l2, missing_ids = refine_chunk_index_skeleton(
        sk_ch,
        l2dict_r,
        cv=cv,
        refine_inds=refine_inds,
        scale_chunk_index=True,
        root_location=root_point_euc,
        nan_rounds=nan_rounds,
        return_missing_ids=True,
        segmentation_fallback=segmentation_fallback,
        fallback_mip=fallback_mip,
        cache=cache,
        save_to_cache=save_to_cache,
        client=client,
        l2cache=l2cache,
    )

    if refine == "chunk" or refine is None:
        refinement_method = None
    elif segmentation_fallback:
        refinement_method = "mesh_average_with_seg_fallback"
    else:
        refinement_method = "mesh_average"
    metameta = {
        "refinement": refine,
        "refinement_method": refinement_method,
        "nan_rounds": nan_rounds,
    }
    try:
        sk_l2.meta.update_metameta(metameta)
    except:
        pass

    if collapse_soma and root_point is not None:
        sk_l2 = collapse_pcg_skeleton(
            sk_l2.vertices[sk_l2.root], sk_l2, collapse_radius
        )

    if refine == "chunk":
        sk_l2._rooted._vertices = utils.nm_to_chunk(sk_l2.vertices, cv)
        try:
            sk_l2.meta.update_metameta({"space": "chunk"})
        except:
            pass

    output = [sk_l2]
    if return_mesh:
        output.append(mesh_ch)
    if return_l2dict:
        output.append((sk_utils.propagate_l2dict(sk_l2, l2dict_mesh), l2dict_r))
    if return_l2dict_mesh:
        output.append((l2dict_mesh, l2dict_mesh_r))
    if return_missing_ids:
        output.append(missing_ids)
    if len(output) == 1:
        return output[0]
    else:
        return tuple(output)


def pcg_meshwork(
    root_id,
    datastack_name=None,
    client=None,
    cv=None,
    refine="all",
    root_point=None,
    root_point_resolution=None,
    root_point_search_radius=300,
    collapse_soma=False,
    collapse_radius=DEFAULT_COLLAPSE_RADIUS,
    synapses=None,
    synapse_table=None,
    remove_self_synapse=True,
    live_query=False,
    timestamp=None,
    invalidation_d=3,
    segmentation_fallback=False,
    fallback_mip=2,
    cache=None,
    save_to_cache=False,
    n_parallel=None,
    l2cache=False,
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
    refine : 'all', 'ep', 'bp', 'epbp'/'bpep', or None, optional
        Selects how to refine vertex locations by downloading mesh chunks.
        Unrefined vertices are placed in the center of their chunk in euclidean space.
        * 'all' refines all vertex locations. (Default)
        * 'ep' refines end points only
        * 'bp' refines branch points only
        * 'bpep' or 'epbp' refines both branch and end points.
        * 'chunk' keeps vertices in chunk index space.
        * None refines no points but maps them to the center of the chunk in euclidean space.
    root_point : array-like or None, optional
        3 element xyz location for the location to set the root in units set by root_point_resolution,
        by default None. If None, a distal tip is selected.
    root_point_resolution : array-like, optional
        Resolution in euclidean space of the root_point, by default [4, 4, 40]
    root_point_search_radius : int, optional
        Distance in euclidean space to look for segmentation when finding the root vertex, by default 300
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
    invalidation_d : int, optional
        Invalidation radius in hops for the mesh skeletonization along the chunk adjacency graph, by default 3
    cache : str or None, optional
        If set to 'service', uses the online l2cache service (if available). Otherwise, this is the filename of a sqlite database with cached lookups for l2 ids. Optional, default is None.
    n_parallel : int, optional
        Number of parallel downloads passed to cloudvolume, by default 1
    l2cache : bool, optional
        Set to True to use the l2cache. Equivalent to cache='service'.

    Returns
    -------
    meshparty.meshwork.Meshwork
        Meshwork object with skeleton based on the level 2 graph. See documentation for details.
    """

    if client is None:
        client = CAVEclient(datastack_name)
    if n_parallel is None:
        n_parallel = 1
    if cv is None:
        cv = client.info.segmentation_cloudvolume(progress=True, parallel=1)
    if root_point_resolution is None:
        root_point_resolution = cv.mip_resolution(0)

    sk_l2, mesh_chunk, (l2dict_mesh, l2dict_mesh_r) = pcg_skeleton(
        root_id,
        client=client,
        cv=cv,
        root_point=root_point,
        root_point_resolution=root_point_resolution,
        root_point_search_radius=root_point_search_radius,
        collapse_soma=collapse_soma,
        collapse_radius=collapse_radius,
        refine=refine,
        invalidation_d=invalidation_d,
        n_parallel=n_parallel,
        return_mesh=True,
        return_l2dict_mesh=True,
        segmentation_fallback=segmentation_fallback,
        fallback_mip=fallback_mip,
        cache=cache,
        save_to_cache=save_to_cache,
        l2cache=l2cache,
    )

    nrn = meshwork.Meshwork(mesh_chunk, seg_id=root_id, skeleton=sk_l2)

    if synapses is not None and synapse_table is not None:
        if synapses == "pre":
            pre, post = True, False
        elif synapses == "post":
            pre, post = False, True
        elif synapses == "all":
            pre, post = True, True
        else:
            raise ValueError('Synapses must be one of "pre", "post", or "all".')

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
        )

    features.add_lvl2_ids(nrn, l2dict_mesh)

    if refine != "chunk":
        chunk_tools.adjust_meshwork(nrn, cv)

    return nrn
