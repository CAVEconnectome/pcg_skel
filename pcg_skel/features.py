import pandas as pd
import numpy as np
from . import pcg_anno
from meshparty.meshwork.algorithms import strahler_order, split_axon_by_annotation

VOL_PROPERTIES = ["area_nm2", "size_nm3", "mean_dt_nm", "max_dt_nm"]


def add_synapses(
    nrn,
    synapse_table,
    l2dict_mesh,
    client,
    root_id=None,
    pre=False,
    post=False,
    remove_self_synapse=True,
    timestamp=None,
    live_query=False,
):
    """Add synapses based on l2ids

    Parameters
    ----------
    nrn : _type_
        _description_
    synapse_table : _type_
        _description_
    l2dict_mesh : _type_
        _description_
    client : _type_
        _description_
    root_id : _type_, optional
        _description_, by default None
    pre : bool, optional
        _description_, by default False
    post : bool, optional
        _description_, by default False
    remove_self_synapse : bool, optional
        _description_, by default True
    timestamp : _type_, optional
        _description_, by default None
    live_query : bool, optional
        _description_, by default False
    """
    if root_id is None:
        root_id - nrn.seg_id

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
    )

    if pre_syn_df is not None:
        nrn.anno.add_annotations(
            "pre_syn",
            pre_syn_df,
            index_column="pre_pt_mesh_ind",
            point_column="ctr_pt_position",
        )
    if post_syn_df is not None:
        nrn.anno.add_annotations(
            "post_syn",
            post_syn_df,
            index_column="post_pt_mesh_ind",
            point_column="ctr_pt_position",
        )

    return


def add_lvl2_ids(
    nrn,
    l2dict_mesh,
):
    lvl2_df = pd.DataFrame(
        {"lvl2_id": list(l2dict_mesh.keys()), "mesh_ind": list(l2dict_mesh.values())}
    )
    nrn.anno.add_annotations("lvl2_ids", lvl2_df, index_column="mesh_ind")
    return


def add_volumetric_properties(
    nrn,
    client,
    attributes=VOL_PROPERTIES,
    l2id_anno_name="lvl2_ids",
    l2id_col_name="lvl2_id",
    property_name="vol_prop",
):
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

    return


def add_segment_properties(
    nrn,
    segment_property_name="segment_properties",
    effective_radius=True,
    area_factor=True,
    strahler=True,
    strahler_by_compartment=False,
    volume_property_name="vol_prop",
    volume_col_name="size_nm3",
    area_col_name="area_nm2",
    root_as_sphere=True,
    comp_mask="is_axon",
):
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
    return


def add_is_axon_annotation(
    nrn,
    pre_anno,
    post_anno,
    annotation_name="is_axon",
    threshold_quality=0.5,
    extend_to_segment=True,
    n_times=1,
):
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
        raise Warning("Split quality below threshold, no axon mesh vertices added!")
    else:
        nrn.anno.add_annotations(annotation_name, is_axon, mask=True)
    return


def l2dict_from_anno(
    nrn,
    table_name="lvl2_ids",
    l2id_col="lvl2_id",
    mesh_ind_col="mesh_ind",
):
    return nrn.anno[table_name].df.set_index(l2id_col)[mesh_ind_col].to_dict()