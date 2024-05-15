import cloudvolume
import fastremap
import multiwrapper.multiprocessing_utils as mu
import numpy as np
from scipy import spatial

from . import chunk_cache, utils

UnshardedMeshSource = (
    cloudvolume.datasource.graphene.mesh.unsharded.GrapheneUnshardedMeshSource
)
ShardedMeshSource = (
    cloudvolume.datasource.graphene.mesh.sharded.GrapheneShardedMeshSource
)

L2_SERVICE_NAME = "service"


class CompleteDataException(Exception):
    pass

def build_graph_topology(lvl2_edge_graph):
    """Extract the graph from the result of the lvl2_graph endpoint"""
    lvl2_edge_graph = np.unique(np.sort(lvl2_edge_graph, axis=1), axis=0)
    lvl2_ids = np.unique(lvl2_edge_graph)
    l2dict = {l2id: ii for ii, l2id in enumerate(lvl2_ids)}
    eg_arr_rm = fastremap.remap(lvl2_edge_graph, l2dict)
    l2dict_reversed = {ii: l2id for l2id, ii in l2dict.items()}
    return eg_arr_rm, l2dict, l2dict_reversed, lvl2_ids


def build_spatial_graph(
    lvl2_edge_graph, cv, client=None, method="chunk", require_complete=False
):
    """Extract spatial graph and level 2 id lookups from chunkedgraph "lvl2_graph" endpoint.

    Parameters
    ----------
    lvl2_edge_graph : array
        Nx2 edge list of level 2 ids
    cv : cloudvolume.CloudVolume
        Associated cloudvolume object

    Returns
    -------
    eg_arr_rm : np.array
        Nx2 edge list of indices remapped to integers starting at 0 through M, the number of unique level 2 ids.
    l2dict : dict
        Dict with level 2 ids as keys and vertex index as values
    l2dict_reversed : dict
        Dict with vertex index as keys and level 2 id as values
    x_ch : np.array
        Mx3 array of vertex locations in chunk index space.
    """
    eg_arr_rm, l2dict, l2dict_reversed, lvl2_ids = build_graph_topology(lvl2_edge_graph)
    if method == "chunk":
        x_ch = [np.array(cv.mesh.meta.meta.decode_chunk_position(l)) for l in lvl2_ids]
    elif method == "service":
        x_ch = dense_spatial_lookup(
            lvl2_ids,
            eg_arr_rm,
            client,
            require_complete=require_complete,
        )
    return eg_arr_rm, l2dict, l2dict_reversed, x_ch

def adjust_meshwork(nrn, cv):
    """Transform vertices in chunk index space to euclidean"""
    nrn._mesh.vertices = utils.chunk_to_nm(nrn._mesh.vertices, cv)


def dense_spatial_lookup(l2ids, eg, client, require_complete=False):
    l2means = np.full((len(l2ids), 3), np.nan)
    locs, inds_found = chunk_cache.get_locs_remote(l2ids, client)
    if require_complete:
        if not np.all(inds_found):
            raise CompleteDataException("Some chunk indices are not yet computed")
    l2means[inds_found] = locs
    return l2means


def refine_vertices(
    vertices,
    l2dict_reversed,
    cv,
    refine_inds="all",
    scale_chunk_index=True,
    convert_missing=False,
    return_missing_ids=True,
    cache=None,
    save_to_cache=False,
    segmentation_fallback=True,
    fallback_mip=2,
    client=None,
):
    """Refine vertices in chunk index space by converting to euclidean space using a combination of mesh downloading and simple chunk mapping.

    Parameters
    ----------
    vertices : array
        Nx3 array of vertex locations in chunk index space
    l2dict_reversed : dict or array
        N-length mapping from vertex index to uint64 level 2 id.
    cv : cloudvolume.CloudVolume
        CloudVolume associated with the chunkedgraph instance
    refine_inds : array, string, or None, optional
        Array of indices to refine via mesh download and recentering, None, or 'all'. If 'all', does all vertices. By default 'all'.
    scale_chunk_index : bool, optional
        If True, moves chunk indices to the euclidean space (the center of the chunk) if not refined. by default True.
    convert_missing : bool, optional
        If True, vertices with missing meshes are converted to the center of their chunk. Otherwise, they are given nans. By default, False.
    return_missing_ids : bool, optional
        If True, returns a list of level 2 ids for which meshes were not found, by default True
    segmentation_fallback : bool, optional
        If True, uses the segmentation as a fallback if the mesh does not exist.

    Returns
    -------
    new_vertices : array
        Nx3 array of remapped vertex locations in euclidean space
    missing_ids : array, optional
        List of level 2 ids without meshes. Only returned if return_missing_ids is True.
    """
    vertices = vertices.copy()

    if isinstance(refine_inds, str):
        if refine_inds == "all":
            refine_inds = np.arange(0, len(vertices))
        else:
            raise ValueError("Invalid value for refine_inds")

    if refine_inds is not None:
        l2ids = [l2dict_reversed[k] for k in refine_inds]
        pt_locs, missing_ids = lvl2_fragment_locs(
            l2ids,
            cv,
            return_missing=True,
            cache=cache,
            save_to_cache=save_to_cache,
            segmentation_fallback=segmentation_fallback,
            fallback_mip=fallback_mip,
            client=client,
        )

        if convert_missing:
            missing_inds = np.any(np.isnan(pt_locs), axis=1)
            vertices[refine_inds[~missing_inds]] = pt_locs[~missing_inds]
        else:
            missing_inds = np.full(len(pt_locs), False)
            vertices[refine_inds] = pt_locs
    else:
        refine_inds = np.array([], dtype=int)
        missing_inds = np.array([], dtype=bool)
        missing_ids = np.array([], dtype=int)

    if scale_chunk_index and len(refine_inds) != len(vertices):
        # Move unrefined vertices to center of chunks
        other_inds = np.full(len(vertices), True)
        if len(refine_inds) > 0:
            other_inds[refine_inds[~missing_inds]] = False
        vertices[other_inds] = (
            utils.chunk_to_nm(vertices[other_inds], cv) + utils.chunk_dims(cv) // 2
        )
    if return_missing_ids:
        return vertices, missing_ids
    else:
        return vertices


def get_root_id_from_point(point, voxel_resolution, client):
    cv = cloudvolume.CloudVolume(
        client.info.segmentation_source(),
        use_https=True,
        bounded=False,
        fill_missing=True,
        progress=False,
        secrets={"token": client.auth.token},
    )
    return int(
        cv.download_point(
            point, size=1, coord_resolution=voxel_resolution, agglomerate=True
        )
    )


def get_closest_lvl2_chunk(
    point,
    root_id,
    client,
    cv=None,
    voxel_resolution=None,
    radius=200,
    return_point=False,
):
    """Get the closest level 2 chunk on a root id

    Parameters
    ----------
    point : array-like
        Point in voxel space.
    root_id : int
        Root id of the object
    client : CAVEclient
        CAVE client to access data
    cv : cloudvolume.CloudVolume or None, optional
        Cloudvolume associated with the dataset. One is created if None.
    voxel_resolution : list, optional
        Point resolution to map between point resolution and mesh resolution, by default [4, 4, 40]
    radius : int, optional
        Max distance to look for a nearby supervoxel. Optional, default is 200.
    return_point : bool, optional
        If True, returns the closest point in addition to the level 2 id. Optional, default is False.

    Returns
    -------
    level2_id : int
        Level 2 id of the object nearest to the point specified.
    close_point : array, optional
        Closest point inside the object to the specified point. Only returned if return_point is True.
    """
    if cv is None:
        cv = cloudvolume.CloudVolume(
            client.info.segmentation_source(),
            use_https=True,
            bounded=False,
            fill_missing=True,
            progress=False,
            secrets={"token": client.auth.token},
        )

    if voxel_resolution is None:
        voxel_resolution = np.array(cv.mip_resolution(0))

    point = point * np.array(voxel_resolution)
    # Get the closest adjacent point for the root id within the radius.
    mip_scaling = np.array(cv.mip_resolution(0))

    pt = np.array(np.array(point) // mip_scaling, np.int32)
    offset = np.array(radius // mip_scaling, np.int32)
    lx = np.array(pt) - offset
    ux = np.array(pt) + offset
    bbox = cloudvolume.Bbox(lx.astype(int), ux.astype(int))
    vol = cv.download(bbox, segids=[root_id])
    vol = np.squeeze(vol)
    if not bool(np.any(vol > 0)):
        raise ValueError("No point of the root id is near the specified point")

    ctr = offset * mip_scaling  # Offset is at center of the volume.
    xyz = np.vstack(np.where(vol > 0)).T
    xyz_nm = xyz * mip_scaling

    ind = np.argmin(np.linalg.norm(xyz_nm - ctr, axis=1))
    closest_pt = vol.bounds.minpt + xyz[ind]

    # Look up the level 2 supervoxel for that id.
    closest_sv = int(cv.download_point(closest_pt, size=1))
    lvl2_id = client.chunkedgraph.get_root_id(closest_sv, level2=True)

    if return_point:
        return lvl2_id, closest_pt * mip_scaling
    else:
        return lvl2_id


def lvl2_fragment_locs(
    l2_ids,
    cv=None,
    return_missing=True,
    segmentation_fallback=True,
    fallback_mip=2,
    cache=None,
    save_to_cache=False,
    client=None,
):
    """Look up representitive location for a list of level 2 ids.

    The representitive point for a mesh is the mesh vertex nearest to the
    centroid of the mesh fragment.

    Parameters
    ----------
    l2_ids : list-like
        List of N level 2 ids
    cv : cloudvolume.CloudVolume
        Associated cloudvolume object
    return_missing : bool, optional
        If True, returns ids of missing meshes. Default is True
    segmentation_fallback : bool, optional
        If True, uses segmentation to get the location if no mesh fragment is found.
        This is slower, but more robust. Default is True.
    cache: str or None, optional
        If a string, filename for a sqlite database used as a lookup cache for l2 ids.
        Default is None.
    save_to_cache: bool, optional
        If True and a chace is set, automatically saves looked up locations to the cache.
        Default is False.
    client : CAVEclient, optional
        Client to be used if the remote service is being used.

    Returns
    -------
    l2means : np.array
        Nx3 list of point locations. Missing mesh fragments get a nan for each component.
    missing_ids : np.array
        List of level 2 ids that were not found. Only returned if return_missing is True.
    """

    l2_ids = np.array(l2_ids)
    l2means = np.full((len(l2_ids), 3), np.nan, dtype=float)
    if cache is not None:
        if cache == L2_SERVICE_NAME:
            l2means_cached, is_cached = chunk_cache.lookup_cached_ids(
                l2_ids, remote_cache=True, client=client
            )
        else:
            l2means_cached, is_cached = chunk_cache.lookup_cached_ids(
                l2_ids, remote_cache=False, cache_file=cache
            )
    else:
        l2means_cached = np.zeros((0, 3), dtype=float)
        is_cached = np.full(len(l2_ids), False, dtype=np.bool)

    l2means[is_cached] = l2means_cached

    if np.any(~is_cached):
        l2means_dl, missing_ids = download_lvl2_locs(
            l2_ids[~is_cached], cv, segmentation_fallback, fallback_mip
        )
        l2means[~is_cached] = l2means_dl
        if cache is not None and save_to_cache and cache != L2_SERVICE_NAME:
            chunk_cache.save_ids_to_cache(l2_ids[~is_cached], l2means_dl, cache)
    else:
        missing_ids = []

    if return_missing:
        return l2means, missing_ids
    else:
        return l2means


def download_lvl2_locs(l2_ids, cv, segmentation_fallback, fallback_mip=2):
    l2meshes = download_l2meshes(
        l2_ids, cv, sharded=isinstance(cv.mesh, ShardedMeshSource)
    )

    l2means = []
    args = []
    auth_token = cv.meta.auth_header["Authorization"][len("Bearer ") :]
    for l2id in l2_ids:
        args.append(
            (
                l2id,
                l2meshes.get(l2id, None),
                f"graphene://{cv.meta.table_path}",
                auth_token,
                segmentation_fallback,
                fallback_mip,
            )
        )

    l2means = mu.multiprocess_func(_localize_l2_id_multi, args, n_threads=1)

    if len(l2means) > 0:
        l2means = np.vstack(l2means)
        missing_ids = l2_ids[np.isnan(l2means[:, 0])]
    else:
        l2means = np.empty((0, 3), dtype=float)
        missing_ids = []
    return l2means, missing_ids


def _localize_l2_id_multi(args):
    l2id, l2mesh, cv_path, auth_token, segmentation_fallback, fallback_mip = args
    return _localize_l2_id(
        l2id, l2mesh, cv_path, auth_token, segmentation_fallback, fallback_mip
    )


def _localize_l2_id(
    l2id, l2mesh, cv_path, auth_token, segmentation_fallback, fallback_mip
):
    if l2mesh is not None:
        l2m_abs = np.mean(l2mesh.vertices, axis=0)
        _, ii = spatial.cKDTree(l2mesh.vertices).query(l2m_abs)
        l2m = l2mesh.vertices[ii]
    else:
        if segmentation_fallback:
            cv = cloudvolume.CloudVolume(
                cv_path,
                bounded=False,
                progress=False,
                fill_missing=True,
                use_https=True,
                mip=0,
                secrets={"token": auth_token},
            )
            try:
                l2m = chunk_location_from_segmentation(l2id, cv, mip=fallback_mip)
            except:
                l2m = np.array([np.nan, np.nan, np.nan])
            del cv
        else:
            l2m = np.array([np.nan, np.nan, np.nan])
    return l2m


def download_l2meshes(l2ids, cv, sharded=False):
    cv.parallel = 1
    if sharded:
        return cv.mesh.get_meshes_on_bypass(l2ids, allow_missing=True)
    else:
        return cv.mesh.get(
            l2ids, allow_missing=True, deduplicate_chunk_boundaries=False
        )


def chunk_location_from_segmentation(l2id, cv, mip=0):
    """Representative level 2 id location using the segmentation.
    This is typically slower than the mesh approach, but is more robust.

    Parameters
    ----------
    l2id : int
        Level 2 id to look up
    cv : cloudvolume.CloudVolume
        CloudVolume associated with the mesh

    Returns
    -------
    xyz_rep : np.array
        3-element xyz array of the representative location in euclidean space.
    """
    loc_ch = np.array(cv.mesh.meta.meta.decode_chunk_position(l2id))
    loc_vox = np.atleast_2d(loc_ch) * cv.graph_chunk_size + np.array(cv.voxel_offset)

    bbox = cloudvolume.Bbox(loc_vox[0], loc_vox[0] + cv.graph_chunk_size)
    bbox_mip = cv.bbox_to_mip(bbox, 0, mip)

    try:
        sv_vol, _ = cv.download(bbox_mip, segids=(l2id,), mip=mip, renumber=True)
        sv_vol = sv_vol > 0
        x, y, z = np.where(sv_vol.squeeze())

        if len(x) == 0 and mip > 0:
            # If too small for the requested mip level, jump to the highest mip level
            del sv_vol
            return chunk_location_from_segmentation(l2id, cv, mip=0)
    except:
        return np.array([np.nan, np.nan, np.nan])

    xyz = np.vstack((x, y, z)).T * np.array(sv_vol.resolution)

    xyz_mean = np.mean(xyz, axis=0)
    xyz_box = xyz[np.argmin(np.linalg.norm(xyz - xyz_mean, axis=1))]
    xyz_rep = np.array(sv_vol.bounds.minpt) * np.array(sv_vol.resolution) + xyz_box
    del sv_vol
    return xyz_rep
