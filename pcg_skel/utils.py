import numpy as np
from trimesh import creation
from functools import reduce


def chunk_dims(cv):
    """Gets the size of a chunk in euclidean space

    Parameters
    ----------
    cv : cloudvolume.CloudVolume
        Chunkedgraph-targeted cloudvolume object

    Returns
    -------
    np.array
        3-element box dimensions of a chunk in nanometers.
    """
    dims = chunk_to_nm([1, 1, 1], cv) - chunk_to_nm([0, 0, 0], cv)
    return np.squeeze(dims)


def nm_to_chunk(xyz_nm, cv, voxel_resolution=[4, 4, 40]):
    """Map a location in euclidean space to a chunk

    Parameters
    ----------
    xyz_nm : array-like
        Nx3 array of spatial points
    cv : cloudvolume.CloudVolume
        CloudVolume object associated with the chunked space 
    voxel_resolution : list, optional
        Voxel resolution, by default [4, 4, 40]

    Returns
    -------
    np.array
        Nx3 array of chunk indices
    """
    mip_scaling = cv.mip_resolution(
        0) // np.array(voxel_resolution, dtype=int)
    x_vox = np.atleast_2d(xyz_nm) / (np.array(mip_scaling)
                                     * np.array(voxel_resolution))
    offset_vox = np.array(cv.mesh.meta.meta.voxel_offset(0))
    return (x_vox + offset_vox) / np.array(cv.mesh.meta.meta.graph_chunk_size)


def chunk_to_nm(xyz_ch, cv, voxel_resolution=[4, 4, 40]):
    """Map a chunk location to Euclidean space

    Parameters
    ----------
    xyz_ch : array-like
        Nx3 array of chunk indices
    cv : cloudvolume.CloudVolume
        CloudVolume object associated with the chunked space 
    voxel_resolution : list, optional
        Voxel resolution, by default [4, 4, 40]

    Returns
    -------
    np.array
        Nx3 array of spatial points
    """
    mip_scaling = cv.mip_resolution(
        0) // np.array(voxel_resolution, dtype=int)

    x_vox = np.atleast_2d(xyz_ch) * cv.mesh.meta.meta.graph_chunk_size
    return (
        (x_vox + np.array(cv.mesh.meta.meta.voxel_offset(0)))
        * voxel_resolution
        * mip_scaling
    )


def _tmat(xyz):
    """4x4 transformation matrix for simple translation by xyz vector"""
    T = np.eye(4)
    T[0:3, 3] = np.array(xyz).reshape(1, 3)
    return T


def chunk_box(xyz, chunk_size=[1, 1, 1]):
    """Create a trimesh box of a specified size

    Parameters
    ----------
    xyz : array-like
        3-element array for the lower corner of the box
    chunk_size : array-like, optional
        3-element array giving box dimensions, by default [1, 1, 1]

    Returns
    -------
    trimesh.mesh
    """
    xyz_offset = xyz + np.array(chunk_size) / 2
    return creation.box(chunk_size, _tmat(xyz_offset))


def chunk_mesh(xyz_ch, cv):
    """Get a trimesh mesh of a collection of chunks.

    Parameters
    ----------
    xyz_ch : np.array
        Nx3 array of chunk indices
    cv : cloudvolume.CloudVolume
        Home cloudvolume object for the chunks.

    Returns
    -------
    trimesh.mesh
    """
    verts_ch_nm = chunk_to_nm(xyz_ch, cv)
    dim = chunk_dims(cv)

    boxes = [chunk_box(xyz, dim) for xyz in verts_ch_nm]
    boxes_all = reduce(lambda a, b: a+b, boxes)
    return boxes_all
