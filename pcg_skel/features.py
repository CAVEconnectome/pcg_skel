import numpy as np
import pandas as pd

import caveclient
import datetime
from meshparty import meshwork
from meshparty.meshwork.algorithms import split_axon_by_annotation, strahler_order
from warnings import warn
from typing import Optional

from . import pcg_anno

VOL_PROPERTIES = ["area_nm2", "size_nm3", "mean_dt_nm", "max_dt_nm"]


def add_synapse_count(
    nrn: meshwork.Meshwork,
    anno_name: str = "synapse_count",
    pre_syn: str = "pre_syn",
    post_syn: str = "post_syn",
    pre_syn_index: str = "pre_pt_mesh_ind",
    post_syn_index: str = "post_pt_mesh_ind",
    aggregate_size: bool = True,
):
    """
    Create a synapse count label across mesh vertices from existing synapse annotation.

    This function adds a synapse count annotation to the mesh vertices of a neuron. It aggregates
    synapse data from pre-synaptic and post-synaptic annotations and optionally calculates the mean
    size of the synapses.

    Parameters
    ----------
    nrn : meshwork.Meshwork
        The neuron meshwork object containing the mesh and annotations.
    anno_name : str, optional
        The name of the annotation table to be created, by default "synapse_count".
    pre_syn : str, optional
        The name of the pre-synaptic annotation table, by default "pre_syn".
        If set to None, presynaptic points will be skipped.
    post_syn : str, optional
        The name of the post-synaptic annotation table, by default "post_syn".
        If set to None, postsynaptic points will be skipped
    pre_syn_index : str, optional
        The mesh index column for pre-synaptic annotation, by default "pre_pt_mesh_ind".
    post_syn_index : str, optional
        The mesh index column for post-synaptic annotation, by default "post_pt_mesh_ind".
    aggregate_size : bool, optional
        Whether to aggregate the size of synapses, by default True.

    Returns
    -------
    None
        The function modifies the neuron meshwork object in place by adding the synapse count annotation.
    """

    with nrn.mask_context(np.ones(nrn.mesh.unmasked_size).astype(bool)):
        syn_count_df = pd.DataFrame(index=nrn.mesh_indices)

        if pre_syn:
            syn_out_agg_kwargs = {
                "num_syn_out": pd.NamedAgg("id", "count"),
            }
            if aggregate_size:
                syn_out_agg_kwargs["net_size_out"] = pd.NamedAgg("size", "sum")

            syn_out_df = (
                nrn.anno[pre_syn].df.groupby(pre_syn_index).agg(**syn_out_agg_kwargs)
            )

        if post_syn:
            syn_in_agg_kwargs = {
                "num_syn_in": pd.NamedAgg("id", "count"),
            }
            if aggregate_size:
                syn_in_agg_kwargs["net_size_in"] = pd.NamedAgg("size", "sum")

            syn_in_df = (
                nrn.anno[post_syn].df.groupby(post_syn_index).agg(**syn_in_agg_kwargs)
            )

        syn_count_df = (
            syn_count_df.merge(
                syn_out_df,
                left_index=True,
                right_index=True,
                how="left",
            )
            .merge(
                syn_in_df,
                left_index=True,
                right_index=True,
                how="left",
            )
            .fillna(0)
        )

        if pre_syn:
            syn_count_df["num_syn_out"] = syn_count_df["num_syn_out"].astype(int)
        if post_syn:
            syn_count_df["num_syn_in"] = syn_count_df["num_syn_in"].astype(int)
        syn_count_df = syn_count_df.reset_index().rename(
            columns={"index": "mesh_index"}
        )

        nrn.anno.add_annotations(
            anno_name,
            syn_count_df,
            index_column="mesh_index",
        )
    pass


def add_synapses(
    nrn: meshwork.Meshwork,
    synapse_table: str,
    l2dict_mesh: dict,
    client: caveclient.CAVEclient,
    root_id: int = None,
    pre: bool = False,
    post: bool = False,
    remove_self_synapse: bool = True,
    timestamp: datetime.datetime = None,
    live_query: bool = False,
    metadata: bool = False,
    synapse_partners: bool = False,
    synapse_point_resolution: list[float] = None,
    synapse_representative_point_pre: str = "ctr_pt_position",
    synapse_representative_point_post: str = "ctr_pt_position",
    synapse_reference_tables: dict = {},
) -> None:
    """Add synapses to a meshwork object based on l2ids

    Parameters
    ----------
    nrn : meshparty.meshwork.Meshwork
        Meshwork object
    synapse_table : str
        Annotation table to use for synapses
    l2dict_mesh : dict
        Dictionary mapping l2ids to vertex ids
    client : caveclient.CAVEclient
        Caveclient to use to get annotations
    root_id : int, optional
        Root id of the cell, by default None. If none, uses the root id set in nrn.seg_id.
    pre : bool, optional
        If True, add presynaptic synpases (i.e. outputs), by default False
    post : bool, optional
        If True, add postsynaptic synapses (i.e. inputs), by default False
    remove_self_synapse : bool, optional
        If True, omit synapses where the pre and post root ids are the same, by default True
    timestamp : datetime.datetime, optional
        Datetime to use for root id lookups if not using a materialized version, by default None
    live_query : bool, optional
        If True, use a timestamp to look up root ids, by default False
    synapse_partners : bool, optional
        If True, returns the root id of synapses partners in the dataframe.
        By default, this is False because partner root ids change with editing and are not specified by this cell's data alone.
    """
    if root_id is None:
        root_id = nrn.seg_id

    pre_syn_df, post_syn_df = pcg_anno.get_level2_synapses(
        root_id,
        l2dict_mesh,
        client,
        synapse_table,
        remove_self=remove_self_synapse,
        pre=pre,
        post=post,
        live_query=live_query,
        timestamp=timestamp,
        metadata=metadata,
        synapse_point_resolution=synapse_point_resolution,
        synapse_reference_tables=synapse_reference_tables,
    )

    if pre_syn_df is not None:
        if not synapse_partners:
            pre_syn_df = pre_syn_df.drop(columns=["post_pt_root_id"])
        nrn.anno.add_annotations(
            "pre_syn",
            pre_syn_df,
            index_column="pre_pt_mesh_ind",
            point_column=synapse_representative_point_pre,
            voxel_resolution=pre_syn_df.attrs.get("dataframe_resolution"),
        )
    if post_syn_df is not None:
        if not synapse_partners:
            post_syn_df = post_syn_df.drop(columns=["pre_pt_root_id"])
        nrn.anno.add_annotations(
            "post_syn",
            post_syn_df,
            index_column="post_pt_mesh_ind",
            point_column=synapse_representative_point_post,
            voxel_resolution=post_syn_df.attrs.get("dataframe_resolution"),
        )


def add_lvl2_ids(
    nrn: meshwork.Meshwork,
    l2dict_mesh: dict,
    property_name: str = "lvl2_ids",
) -> None:
    """Add meshwork annotation table associating level 2 ids with vertex ids.

    Parameters
    ----------
    nrn : meshparty.meshwork.Meshwork
        Meshwork object
    l2dict_mesh : dict
        Dictionary mapping L2 ids to mesh graph indices.
    property_name : str, optional
        Name of the annotation table, by default "lvl2_ids"
    """
    lvl2_df = pd.DataFrame(
        {"lvl2_id": list(l2dict_mesh.keys()), "mesh_ind": list(l2dict_mesh.values())}
    )
    nrn.anno.add_annotations(property_name, lvl2_df, index_column="mesh_ind")


def add_volumetric_properties(
    nrn: meshwork.Meshwork,
    client: caveclient.CAVEclient,
    attributes: list[str] = VOL_PROPERTIES,
    l2id_anno_name: str = "lvl2_ids",
    l2id_col_name: str = "lvl2_id",
    property_name: str = "vol_prop",
) -> None:
    """Add L2 Cache properties as an annotation property table.

    Parameters
    ----------
    nrn : meshparty.meshwork.Meshwork
        Meshwork object
    client : caveclient.CAVEclient
        Initialized caveclient
    attributes : list, optional
        List of attributes to download, by default: ["area_nm2", "size_nm3", "mean_dt_nm", "max_dt_nm"].
    l2id_anno_name : str, optional
        Name of the annotation property table holding L2 ids, by default "lvl2_ids"
    l2id_col_name : str, optional
        Name of the column in the property table holding L2 ids, by default "lvl2_id"
    property_name : str, optional
        Name of the new volume property table, by default "vol_prop"
    """
    l2ids = nrn.anno[l2id_anno_name].df[l2id_col_name]
    dat = client.l2cache.get_l2data(l2ids, attributes=attributes)
    dat_df = pd.DataFrame.from_dict(dat, orient="index")
    dat_df.index = [int(x) for x in dat_df.index]

    l2_df = nrn.anno.lvl2_ids.df
    nrn.anno.add_annotations(
        property_name,
        data=l2_df.merge(dat_df, left_on=l2id_col_name, right_index=True).drop(
            columns=[l2id_col_name]
        ),
        index_column="mesh_ind",
    )


def add_segment_properties(
    nrn: meshwork.Meshwork,
    segment_property_name: str = "segment_properties",
    effective_radius: bool = True,
    area_factor: bool = True,
    strahler: bool = True,
    strahler_by_compartment: bool = False,
    volume_property_name: str = "vol_prop",
    volume_col_name: str = "size_nm3",
    area_col_name: str = "area_nm2",
    root_as_sphere: bool = True,
    comp_mask: str = "is_axon",
) -> None:
    """Use volumetric and topological properties to add descriptive properties for each skeleton vertex.
    Note that properties are estimated per segment, the unbranched region between branch points and/or endpoints.

    This function assumes that the volume properties have already been added to the
    meshwork, which can be done using `add_volumetric_properties`.

    Parameters
    ----------
    nrn : meshparty.meshwork.Meshwork
        Meshwork object
    segment_property_name : str, optional
        Name of the new annotation property table, by default "segment_properties"
    effective_radius : bool, optional
        If True, add a column for a radius estimate found by computing the radius of an equivalent cylindar with the same length and volume as each segment, by default True
    area_factor : bool, optional
        If True, add a column with the ratio of the surface area of the segment and the surface area of an equivalent cylinder (without endcaps), by default True
    strahler : bool, optional
        If True, add a column with the strahler number of the segment, by default True
    strahler_by_compartment : bool, optional
        If True, computer strahler number for axon and dendrite separately, using the annotation property table specified in `comp_mask`, by default False
    volume_property_name : str, optional
        Name of the volume properties table as generated by the function `add_volumetric_properties`, by default "vol_prop"
    volume_col_name : str, optional
        Name of the column holding volume, by default "size_nm3"
    area_col_name : str, optional
        Name of the column holding surface area, by default "area_nm2"
    root_as_sphere : bool, optional
        Treats the root location as a sphere for setting the effective radius, by default True
    comp_mask : str, optional
        Sets the annotation table to mask off for strahler number computation, by default "is_axon".
    """
    if volume_property_name not in nrn.anno.table_names:
        msg = f"Volume property table {volume_property_name} not found in `nrn.anno`. "
        msg += "You may want to run `add_volumetric_properties` on this neuron first "
        msg += "before calling this function."
        raise ValueError(msg)

    seg_num = []
    is_root = []
    segment_index = np.zeros(len(nrn.vertices), dtype=int)
    if strahler:
        if strahler_by_compartment:
            with nrn.mask_context(nrn.anno[comp_mask].mesh_mask) as nrnf:
                so_axon = strahler_order(nrnf)
                so_axon_base = np.full(len(nrnf.mesh_mask), np.nan)
                so_axon_base[nrnf.mesh_mask] = so_axon
            with nrn.mask_context(~nrn.anno[comp_mask].mesh_mask) as nrnf:
                so_dend = strahler_order(nrnf)
                so_dend_base = np.full(len(nrnf.mesh_mask), np.nan)
                so_dend_base[nrnf.mesh_mask] = so_dend
            so = np.where(~np.isnan(so_axon_base), so_axon_base, so_dend_base).astype(
                int
            )
        if not strahler_by_compartment:
            so = strahler_order(nrn)
        seg_strahler = []

    if effective_radius:
        seg_vols = []
        seg_pls = []
    if area_factor:
        seg_areas = []
    prop_df = nrn.anno[volume_property_name].df
    prop_df.set_index("mesh_ind")
    for ii, seg in enumerate(nrn.segments()):
        seg_num.append(ii)
        is_root.append(nrn.root in seg)
        segment_index[seg] = ii
        if effective_radius:
            seg_vols.append(prop_df.loc[seg][volume_col_name].sum())
            if nrn.skeleton.root in seg.to_skel_index:
                seg_pls.append(0)
            else:
                seg_plus = nrn.MeshIndex(
                    np.unique(np.concatenate((seg, nrn.parent_index(seg))))
                )  # Add dist to parent node for path length
                seg_pls.append(nrn.path_length(seg_plus))
        if area_factor:
            seg_areas.append(prop_df.loc[seg][area_col_name].sum())
        if strahler:
            seg_strahler.append(so[seg[0]])

    base_df = pd.DataFrame(
        {
            "seg_num": segment_index,
            "mesh_ind": np.arange(len(segment_index)),
        }
    )
    prop_dict = {"seg_num": seg_num, "is_root": is_root}
    if effective_radius:
        prop_dict["vol"] = seg_vols
        prop_dict["len"] = seg_pls
    if area_factor:
        prop_dict["area"] = seg_areas
    if strahler:
        prop_dict["strahler"] = seg_strahler
    prop_df = pd.DataFrame(prop_dict)
    if effective_radius:
        prop_df["r_eff"] = np.sqrt(prop_df["vol"] / (np.pi * prop_df["len"]))
        if root_as_sphere:
            r_idx = prop_df.query("is_root").index
            prop_df.loc[r_idx, "r_eff"] = (
                prop_df.loc[r_idx, "vol"] * (3 / 4) / np.pi
            ) ** (1 / 3)
    if area_factor and effective_radius:
        prop_df["area_factor"] = prop_df["area"] / (
            2 * np.pi * prop_df["r_eff"] * prop_df["len"]
        )
    base_df = base_df.merge(
        prop_df,
        how="left",
        on="seg_num",
    )
    nrn.anno.add_annotations(
        segment_property_name,
        data=base_df,
        index_column="mesh_ind",
    )


def add_is_axon_annotation(
    nrn: meshwork.Meshwork,
    pre_anno: str,
    post_anno: str,
    annotation_name: str = "is_axon",
    threshold_quality: float = 0.5,
    extend_to_segment: bool = True,
    n_times: int = 1,
    return_quality: bool = False,
):
    """Add an annotation property table specifying which vertices belong to the axon, based on synaptic input and output locations
    For the synapse flow centrality algorithm, see "Quantitative neuroanatomy for connectomics in Drosophila", Schneider-Mizell et al. eLife 2016.

    Parameters
    ----------
    nrn : meshparty.meshwork.Meshwork
        Meshwork object
    pre_anno : str
        Annotation property table name for presynaptic sites (outputs).
    post_anno : str
        Annotation property table name for postsyanptic sites (inputs).
    annotation_name : str, optional
        Name of the new table specifying axon indices, by default "is_axon"
    threshold_quality : float, optional
        Value between 0 and 1 setting the lower limit on input/output segregation quality, by default 0.5. If the segregatation quality is lower than this,
        a table is added but no vertices will be in the table.
    extend_to_segment : bool, optional
        If True, the axon split point will be moved to the most root-ward point on the segment containing the highest synapse flow centrality, by default True
    n_times : int, optional
        The number of times to run axon/dendrite detection in a row, by default 1.
        This should be set to the number of distinct axon branches on a cell, which surprisingly can be more than one even for mouse neurons.
    return_quality : bool, optional
        If True, returns the split quality score (a float between 0-1). Default is False.

    Returns
    -------
    split_quality
        Float between 0-1, with higher numbers indicating a better axon/dendrite segregation.
    """
    is_axon, sq = split_axon_by_annotation(
        nrn,
        pre_anno,
        post_anno,
        return_quality=True,
        extend_to_segment=extend_to_segment,
        n_times=n_times,
    )
    if sq < threshold_quality:
        nrn.anno.add_annotations(annotation_name, [], mask=True)
        warn("Split quality below threshold, no axon mesh vertices added!")
    else:
        nrn.anno.add_annotations(annotation_name, is_axon, mask=True)
    if return_quality:
        return sq
    else:
        pass


def l2dict_from_anno(
    nrn: meshwork.Meshwork,
    table_name: str = "lvl2_ids",
    l2id_col: str = "lvl2_id",
    mesh_ind_col: str = "mesh_ind",
) -> dict:
    return nrn.anno[table_name].df.set_index(l2id_col)[mesh_ind_col].to_dict()


def aggregate_property_to_skeleton(
    nrn: meshwork.Meshwork,
    anno_table: str,
    agg_dict: Optional[dict] = None,
    fill_value: float = 0,
) -> pd.DataFrame:
    """Aggregate a meshwork annotation table to skeletons

    Parameters
    ----------
    nrn : meshwork.Meshwork
        Meshwork object with mesh,skeleton, and annotations
    anno_table : str
        Name of the annotation table
    agg_dict : dict
        Dictionary of column names and aggregation functions that can be used by pandas pd.NamedAgg.
    fill_value : float, optional
        Value to use to fill missing or NaN values, by default 0

    Returns
    -------
    pd.DataFrame
        Dataframe with one row per skeleton index and columns aggregating mesh annotation values
        using the aggregation function specified in the aggregation_dict.
    """
    df = nrn.anno[anno_table].df

    skel_index_column = "skel_index"
    while skel_index_column in df.columns:
        skel_index_column = f"{skel_index_column}_temp"
    df[skel_index_column] = nrn.anno[anno_table].mesh_index.to_skel_index_padded

    aggs = {k: pd.NamedAgg(k, v) for k, v in agg_dict.items()}
    skel_df = df.groupby(skel_index_column).agg(**aggs)
    skel_df = skel_df.reindex(np.arange(0, nrn.skeleton.n_vertices)).fillna(fill_value)
    return skel_df
