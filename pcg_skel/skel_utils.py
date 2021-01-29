import numpy as np


def filter_l2dict(sk_ch, l2dict_mesh_r):
    """Converts a reverse level 2 dict for a mesh into one for a skeleton
    """
    l2dict_sk_r = {ii: l2dict_mesh_r[mind]
                   for ii, mind in enumerate(sk_ch.mesh_index)}
    l2dict_sk = {v: k for v, k in l2dict_sk_r.items()}
    return l2dict_sk, l2dict_sk_r


def propagate_l2dict(sk_ch, l2dict_mesh):
    """Propagate a l2dict for a mesh to an l2dict for a skeleton, with multiple
    level 2 ids potentially mapping to each skeleton index
    """
    l2dict_long = {l2dict_mesh[k]: sk_ch.mesh_to_skel_map[v]
                   for k, v in l2dict_mesh.items()}
    return l2dict_long


def fix_nan_verts(sk, num_rounds=20):
    """ Replace vertices with locations that are nan with mean of neighbor locations
    """
    nr = 0
    while nr < num_rounds:

        nanvinds = np.flatnonzero(np.isnan(sk.vertices[:, 0]))
        new_verts = sk.vertices[nanvinds]

        for ii, vid in enumerate(nanvinds):
            neib_inds = np.array(
                sk.csgraph_binary_undirected[vid, :].todense()).squeeze() > 0
            if len(neib_inds) == 0:
                continue
            if np.any(neib_inds):
                new_verts[ii] = np.nanmean(
                    sk.vertices[neib_inds], axis=0) + np.array([0, 0, 1])
        sk._rooted._vertices[nanvinds] = new_verts
        sk._vertices[nanvinds] = new_verts

        if not np.any(np.isnan(new_verts[:, 0])):
            break

        nr += 1
    else:
        print(f'Could not fix all nans after {num_rounds} rounds')
    pass


def get_centered_mesh(mesh):
    if mesh is None:
        return np.array([np.nan, np.nan, np.nan])
    verts = mesh.vertices
    close_ind = np.argmin(np.linalg.norm(verts-np.mean(verts, axis=0), axis=1))
    return verts[close_ind]


def attach_new_skeleton(nrn, new_skeleton):
    nrn._skeleton = new_skeleton
    nrn._recompute_indices()
    pass