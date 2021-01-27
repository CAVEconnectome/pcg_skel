import time
import cloudvolume
import fastremap
import numpy as np
import pandas as pd
from annotationframeworkclient import FrameworkClient, chunkedgraph, frameworkclient
from meshparty import mesh_filters, skeleton, skeletonize, trimesh_io
from scipy import sparse, spatial

from . import skel_utils as sk_utils
from . import utils

DEFAULT_VOXEL_RESOLUTION = [4, 4, 40]


def get_closest_lvl2_chunk(
    point,
    root_id,
    client,
    cv=None,
    voxel_resolution=[4, 4, 40],
    radius=200,
    return_point=False,
):
    """Get the closest level 2 chunk on a root id

    Parameters
    ----------
    point : array-like
        Point in space.
    root_id : int
        Root id of the object
    client : FrameworkClient
        Framework client to access data
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
            progress=False,
        )

    # Get the closest adjacent point for the root id within the radius.
    mip_scaling = cv.mip_resolution(
        0) // np.array(voxel_resolution, dtype=int)

    pt = np.array(point) // mip_scaling
    offset = radius // (np.array(mip_scaling) * np.array(voxel_resolution))
    lx = np.array(pt) - offset
    ux = np.array(pt) + offset
    bbox = cloudvolume.Bbox(lx, ux)
    vol = cv.download(bbox, segids=[root_id])
    vol = np.squeeze(vol)
    if not bool(np.any(vol > 0)):
        raise ValueError("No point of the root id is near the specified point")

    ctr = offset * point * voxel_resolution
    xyz = np.vstack(np.where(vol > 0)).T
    xyz_nm = xyz * mip_scaling * voxel_resolution

    ind = np.argmin(np.linalg.norm(xyz_nm - ctr, axis=1))
    closest_pt = vol.bounds.minpt + xyz[ind]

    # Look up the level 2 supervoxel for that id.
    closest_sv = int(cv.download_point(closest_pt, size=1))
    lvl2_id = client.chunkedgraph.get_root_id(closest_sv, level2=True)

    if return_point:
        return lvl2_id, closest_pt * mip_scaling * voxel_resolution
    else:
        return lvl2_id


def refine_meshwork_vertices(nrn, cv, lvl2_anno='lvl2_ids', lvl2_col='lvl2_id', nan_rounds=20):
    """Refine the skeleton of a meshwork via level 2 meshes

    Parameters
    ----------
    nrn : meshwork.Meshwork
        Meshwork object with vertices in chunkedgraph index space
    cv : cloudvolume.CloudVolume
        CloudVolume with meshes
    lvl2_anno : str, optional
        Name of annotation with level 2 ids, by default 'lvl2_ids'
    lvl2_col : str, optional
        Column name in annotation with level 2 ids, by default 'lvl2_id'
    nan_rounds : int, optional
        Number of rounds of removing nan vertices, by default 20

    Returns
    -------
    nrn
        Meshwork object with updated skeleton
    """
    lvl2_ids = nrn.anno[lvl2_anno].df.loc[nrn.skeleton.mesh_index][lvl2_col].values
    new_sk = refine_skeleton_vertices(
        nrn.skeleton, cv, lvl2_ids, nan_rounds=nan_rounds)
    return sk_utils.attach_new_skeleton(nrn, new_sk)


def refine_skeleton_vertices(sk, cv, lvl2_ids, nan_rounds=20):
    """Refine a skeleton by downloading level 2 meshes for each vertex

    Parameters
    ----------
    sk : meshparty.skeleton.Skeleton
        Skeleton of level 2 ids with N vertices in chunkedgraph index space
    cv : cloudvolume.CloudVolume
        A cloudvolume object with the level2 ids.
    lvl2_ids : list-like
        List of N level 2 ids in same order as vertices
    nan_rounds : int, optional
        Number of iterations of nan removal, by default 10

    Returns
    -------
    Skeleton
        Skeleton with vertices updated to new locations
    """
    lvl2_meshes = cv.mesh.get_meshes_on_bypass(lvl2_ids, allow_missing=True)

    new_verts = np.vstack(
        [sk_utils.get_centered_mesh(lvl2_meshes.get(k, None))
         for k in lvl2_ids]
    )
    sk._rooted._vertices = new_verts
    sk._vertices = new_verts
    sk_utils.fix_nan_verts(sk, num_rounds=nan_rounds)
    return sk


def collapse_pcg_meshwork(soma_pt, nrn, soma_r):
    """Use soma point vertex and collapse soma as sphere

    Parameters
    ----------
    soma_pt : array
        3-element location of soma center (in nm)
    nrn: meshwork.Meshwork
        Coarse meshwork with skeleton
    soma_r : float
        Soma collapse radius (in nm)

    Returns
    -------
    skeleton
        New skeleton with updated properties
    """
    new_skeleton = collapse_pcg_skeleton(soma_pt, nrn.skeleton, soma_r)
    sk_utils.attach_new_skeleton(nrn, new_skeleton)
    return nrn


def collapse_pcg_skeleton(soma_pt, sk, soma_r):
    """Use soma point vertex and collapse soma as sphere

    Parameters
    ----------
    soma_pt : array
        3-element location of soma center (in nm)
    sk: skeleton.Skeleton
        Coarse skeleton
    soma_r : float
        Soma collapse radius (in nm)

    Returns
    -------
    skeleton
        New skeleton with updated properties
    """
    soma_verts, _ = skeletonize.soma_via_sphere(
        soma_pt, sk.vertices, sk.edges, soma_r)
    min_soma_vert = np.argmin(np.linalg.norm(
        sk.vertices[soma_verts] - soma_pt, axis=1))
    root_vert = soma_verts[min_soma_vert]

    new_v, new_e, new_skel_map, vert_filter, root_ind = skeletonize.collapse_soma_skeleton(soma_verts[soma_verts != root_vert],
                                                                                           soma_pt,
                                                                                           sk.vertices,
                                                                                           sk.edges,
                                                                                           sk.mesh_to_skel_map,
                                                                                           collapse_index=root_vert,
                                                                                           return_soma_ind=True,
                                                                                           return_filter=True)

    new_mesh_index = sk.mesh_index[vert_filter]
    new_skeleton = skeleton.Skeleton(
        new_v, new_e, root=root_ind, mesh_to_skel_map=new_skel_map, mesh_index=new_mesh_index)
    return new_skeleton


def _adjust_meshwork(nrn, cv):
    """Transform vertices in chunk index space to euclidean"""

    nrn._mesh.vertices = utils.chunk_to_nm(nrn._mesh.vertices, cv)
    nrn._skeleton._rooted._vertices = utils.chunk_to_nm(
        nrn._skeleton._rooted.vertices, cv)
    nrn._skeleton._vertices = utils.chunk_to_nm(nrn._skeleton.vertices, cv)


def build_spatial_graph(lvl2_edge_graph, cv):
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
    lvl2_edge_graph = np.unique(np.sort(lvl2_edge_graph, axis=1), axis=0)
    lvl2_ids = np.unique(lvl2_edge_graph)
    l2dict = {l2id: ii for ii, l2id in enumerate(lvl2_ids)}
    eg_arr_rm = fastremap.remap(lvl2_edge_graph, l2dict)
    l2dict_reversed = {ii: l2id for l2id, ii in l2dict.items()}

    x_ch = [np.array(cv.mesh.meta.meta.decode_chunk_position(l))
            for l in lvl2_ids]
    return eg_arr_rm, l2dict, l2dict_reversed, x_ch


def cg_space_skeleton(root_id,
                      client=None,
                      datastack_name=None,
                      cv=None,
                      root_point=None,
                      invalidation_d=3,
                      return_mesh=False,
                      return_l2dict=True,
                      n_parallel=4):
    """Generate a basic skeleton with chunked-graph index vertices.

    Parameters
    ----------
    root_id : np.uint64
        Neuron root id
    client : annotationframeworkclient.FrameworkClient, optional
        FrameworkClient for a datastack, by default None. If None, you must specify a datastack name.
    datastack_name : str, optional
        Datastack name to create a FrameworkClient, by default None. Only used if client is None.
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
        Sets number of parallel threads for cloudvolume, by default 4

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
        client = FrameworkClient(datastack_name)
    if cv is None:
        cv = cloudvolume.CloudVolume(client.info.segmentation_source(), parallel=n_parallel,
                                     use_https=True, progress=False, bounded=False)

    lvl2_eg = client.level2_chunk_graph(root_id)

    eg, l2dict, l2dict_reversed, x_ch = build_spatial_graph(lvl2_eg, cv)
    mesh_chunk = trimesh_io.Mesh(vertices=x_ch, faces=[], link_edges=eg)

    if root_point is not None:
        lvl2_root_chid, lvl2_root_loc = get_closest_lvl2_chunk(
            root_point, root_id, client=client, cv=None, radius=300, return_point=True)  # Need to have cv=None because of a cloudvolume inconsistency
        root_mesh_index = l2dict[lvl2_root_chid]
    else:
        root_mesh_index = None

    sk_ch = skeletonize.skeletonize_mesh(
        mesh_chunk, invalidation_d=invalidation_d,
        collapse_soma=False, compute_radius=False,
        root_index=root_mesh_index, remove_zero_length_edges=False)

    out_list = [sk_ch]
    if return_mesh:
        out_list.append(mesh_chunk)
    if return_l2dict:
        out_list.append(l2dict)
    if len(out_list) == 1:
        return out_list[0]
    else:
        return tuple(out_list)
