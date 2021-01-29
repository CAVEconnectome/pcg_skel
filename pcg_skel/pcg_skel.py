import time
import cloudvolume
import fastremap
import numpy as np
import pandas as pd
from annotationframeworkclient import FrameworkClient, chunkedgraph, frameworkclient
from meshparty import mesh_filters, skeleton, skeletonize, trimesh_io
from scipy import sparse, spatial

from . import chunk_tools
from . import skel_utils as sk_utils
from . import utils

DEFAULT_VOXEL_RESOLUTION = [4, 4, 40]


# def refine_meshwork_vertices(nrn, cv, lvl2_anno='lvl2_ids', lvl2_col='lvl2_id', nan_rounds=20):
#     """Refine the skeleton of a meshwork via level 2 meshes

#     Parameters
#     ----------
#     nrn : meshwork.Meshwork
#         Meshwork object with vertices in chunkedgraph index space
#     cv : cloudvolume.CloudVolume
#         CloudVolume with meshes
#     lvl2_anno : str, optional
#         Name of annotation with level 2 ids, by default 'lvl2_ids'
#     lvl2_col : str, optional
#         Column name in annotation with level 2 ids, by default 'lvl2_id'
#     nan_rounds : int, optional
#         Number of rounds of removing nan vertices, by default 20

#     Returns
#     -------
#     nrn
#         Meshwork object with updated skeleton
#     """
#     lvl2_ids = nrn.anno[lvl2_anno].df.loc[nrn.skeleton.mesh_index][lvl2_col].values
#     new_sk = refine_skeleton_vertices(
#         nrn.skeleton, cv, lvl2_ids, nan_rounds=nan_rounds)
#     return sk_utils.attach_new_skeleton(nrn, new_sk)


# def refine_meshwork(
#     nrn_ch,
#     cv,
#     l2_anno=None,
#     l2_col='lvl2_id',
#     l2dict_reversed=None,
#     refine_inds='skeleton',
#     scale_chunk_index=True,
#     root_location=None,
#     nan_rounds=20,
#     return_missing_ids=False,
# ):
#     if l2_anno is not None:
#         # Build from a meshwork annotation
#         sk_l2_ids = nrn_ch.anno[lvl2_anno].df.loc[nrn_ch.skeleton.mesh_index][lvl2_col].values
#         l2dict_reversed = {ii: k for ii, k in enumerate(sk_l2_ids)}
#     if refine_inds == 'skeleton':
#         refine_inds_sk = 'all'
#     else:

#         # Do the remapping
#     new_sk = refine_skeleton(nrn_ch.skeleton,
#                              l2dict_reversed=l2dict_reversed,
#                              cv=cv,
#                              refine_inds=refine_inds_sk)


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


def chunk_index_skeleton(root_id,
                         client=None,
                         datastack_name=None,
                         cv=None,
                         root_point=None,
                         invalidation_d=3,
                         return_mesh=False,
                         return_l2dict=False,
                         return_mesh_l2dict=False,
                         root_point_resolution=[4, 4, 40],
                         root_point_search_radius=300,
                         n_parallel=1):
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
        client = FrameworkClient(datastack_name)
    if cv is None:
        cv = cloudvolume.CloudVolume(client.info.segmentation_source(), parallel=n_parallel,
                                     use_https=True, progress=False, bounded=False)

    lvl2_eg = client.chunkedgraph.level2_chunk_graph(root_id)

    eg, l2dict_mesh, l2dict_r_mesh, x_ch = build_spatial_graph(
        lvl2_eg, cv)
    mesh_chunk = trimesh_io.Mesh(vertices=x_ch, faces=[], link_edges=eg)

    if root_point is not None:
        lvl2_root_chid, lvl2_root_loc = chunk_tools.get_closest_lvl2_chunk(
            root_point,
            root_id,
            client=client,
            cv=None,
            radius=root_point_search_radius,
            voxel_resolution=root_point_resolution,
            return_point=True)  # Need to have cv=None because of a cloudvolume inconsistency
        root_mesh_index = l2dict_mesh[lvl2_root_chid]
    else:
        root_mesh_index = None

    sk_ch = skeletonize.skeletonize_mesh(
        mesh_chunk, invalidation_d=invalidation_d,
        collapse_soma=False, compute_radius=False,
        root_index=root_mesh_index, remove_zero_length_edges=False)

    l2dict, l2dict_r = sk_utils.filter_l2dict(sk_ch, l2dict_r_mesh)

    out_list = [sk_ch]
    if return_mesh:
        out_list.append(mesh_chunk)
    if return_mesh_l2dict:
        out_list.append((l2dict_mesh, l2dict_r_mesh))
    if return_l2dict:
        out_list.append((l2dict, l2dict_r))
    if len(out_list) == 1:
        return out_list[0]
    else:
        return tuple(out_list)


def refine_chunk_index_skeleton(
    sk_ch,
    l2dict_reversed,
    cv,
    refine_inds='all',
    scale_chunk_index=True,
    root_location=None,
    nan_rounds=20,
    return_missing_ids=False,
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

    Returns
    -------
    meshparty.skeleton.Skeleton
        Skeleton with remapped vertex locations
    """
    if nan_rounds is None:
        convert_missing = True
    else:
        convert_missing = False

    refine_out = chunk_tools.refine_vertices(sk_ch.vertices,
                                             l2dict_reversed=l2dict_reversed,
                                             cv=cv,
                                             refine_inds=refine_inds,
                                             scale_chunk_index=scale_chunk_index,
                                             convert_missing=convert_missing,
                                             return_missing_ids=return_missing_ids)
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
    )

    if refine_inds == 'all':
        sk_utils.fix_nan_verts(l2_sk, num_rounds=nan_rounds)

    if return_missing_ids:
        return l2_sk, missing_ids
    else:
        return l2_sk


def pcg_skeleton(root_id,
                 client=None,
                 datastack_name=None,
                 cv=None,
                 refine='all',
                 root_point=None,
                 invalidation_d=3,
                 return_mesh=False,
                 return_l2dict=True,
                 return_missing_ids=False,
                 nan_rounds=20,
                 n_parallel=1):

    if client is None:
        client = FrameworkClient(datastack_name)

    if cv is None:
        cv = cloudvolume.CloudVolume(client.info.segmentation_source(), parallel=n_parallel,
                                     use_https=True, progress=False, bounded=False)

    sk_ch, mesh_ch, (l2dict, l2dict_reversed) = chunk_index_skeleton(root_id,
                                                                     client=client,
                                                                     datastack_name=datastack_name,
                                                                     cv=cv,
                                                                     root_point=root_point,
                                                                     invalidation_d=invalidation_d,
                                                                     return_mesh=True,
                                                                     return_l2dict=True,
                                                                     n_parallel=n_parallel)

    sk_l2, missing_ids = refine_chunk_index_skeleton(sk_ch,
                                                     l2dict_reversed,
                                                     cv=cv,
                                                     refine_inds=refine,
                                                     scale_chunk_index=True,
                                                     root_location=root_point,
                                                     nan_rounds=nan_rounds,
                                                     return_missing_ids=True)

    output = [sk_l2]
    if return_mesh:
        output.append(mesh_ch)
    if return_l2dict:
        output.append((l2dict, l2dict_reversed))
    if return_missing_ids:
        output.append(missing_ids)
    if len(output) == 1:
        return output[0]
    else:
        return tuple(output)


def lvl2_branch_fragment_locs(sk_ch, lvl2dict_reversed, cv):
    br_minds = sk_ch.mesh_index[sk_ch.branch_points_undirected]
    branch_l2s = list(map(lambda x: lvl2dict_reversed[x], br_minds))
    return chunk_tools.lvl2_fragment_locs(branch_l2s, cv)


def lvl2_end_fragment_locs(sk_ch, lvl2dict_reversed, cv):
    ep_minds = sk_ch.mesh_index[sk_ch.end_points_undirected]
    ep_l2s = list(map(lambda x: lvl2dict_reversed[x], ep_minds))
    return chunk_tools.lvl2_fragment_locs(ep_l2s, cv)
