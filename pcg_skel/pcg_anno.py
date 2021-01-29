import numpy as np
from annotationframeworkclient import frameworkclient


def annotation_to_level2_id(df,
                            client,
                            bound_pt_columns='pt',
                            l2_suffix='_level2_id',
                            sv_columns=None,
                            l2_columns=None,
                            inplace=False):
    """Add or more level2_id columns to a dataframe based on supervoxel columns

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with one or more supervoxel columns
    client : FrameworkClient or ChunkedgraphClient
        Client for interacting with the chunkedgraph
    bound_pt_columns : str or list-like, optional
        List of bound spatial point names, prefix only. E.g. 'pt' for an annotation with
        'pt_position', 'pt_supervoxel_id', and 'pt_root_id'. Optional, by default 'pt'.
    l2_suffix : str, optional
        Suffix to use for new level 2 id column, by default '_level2_id'
    sv_columns : str or list-like, optional
        , by default None
    l2_columns : [type], optional
        [description], by default None
    inplace : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    if isinstance(bound_pt_columns, str):
        bound_pt_columns = [bound_pt_columns]
    if sv_columns is None:
        sv_columns = [f'{c}_supervoxel_id' for c in bound_pt_columns]
        if l2_columns is None:
            l2_columns = [f'{c}{l2_suffix}' for c in bound_pt_columns]
    elif l2_columns is None:
        l2_columns = [f'{c}{l2_suffix}' for c in sv_columns]

    if isinstance(client, frameworkclient.FrameworkClientFull):
        pcg_client = client.chunkedgraph
    else:
        pcg_client = client

    if not inplace:
        df = df.copy()

    for col, l2_col in zip(sv_columns, l2_columns):
        level2_ids = pcg_client.get_roots(df[col].values,
                                          stop_level=2,
                                          timestamp=client.materialize.get_timestamp()
                                          )
        df[l2_col] = level2_ids
    return df


def annotation_to_mesh_index(df,
                             l2dict,
                             level2_id_col='pt_level2_id',
                             mesh_index_col='pt_mesh_ind',
                             inplace=False):
    """Map level2 ids to mesh indices.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with at least one level 2 id column
    l2dict : dict
        Dict of level2 id to mesh index mappings
    level2_id_col : str or list, optional
        Level 2 id column or list of columns, by default 'pt_level2_id'
    mesh_index_col : str, optional
        Column name (or names) to use for added mesh index, by default 'pt_mesh_ind'
    inplace : bool, optional
        If False, makes changes on a copy of the dataframe, by default False

    Returns
    -------
    pandas.DataFrame
        DataFrame with mesh index column/s added
    """
    if not inplace:
        df = df.copy()
    if isinstance(level2_id_col, str):
        level2_id_col = [level2_id_col]
    if isinstance(mesh_index_col, str):
        mesh_index_col = [mesh_index_col]

    for l2col, mind_col in zip(level2_id_col, mesh_index_col):
        df[mind_col] = df[l2col].apply(lambda x: l2dict[x])
    return df


def _mapped_synapses(root_id, client, l2dict, side, synapse_table, remove_self):
    syn_df = client.materialize.query_table(
        synapse_table, filter_equal_dict={f'{side}_pt_root_id': root_id}
    )

    if remove_self:
        syn_df = syn_df.query(
            'pre_pt_root_id != post_pt_root_id').reset_index(drop=True)
    syn_df = annotation_to_level2_id(syn_df,
                                     client,
                                     bound_pt_columns=f'{side}_pt',
                                     inplace=True)
    syn_df = annotation_to_mesh_index(syn_df,
                                      l2dict,
                                      level2_id_col=f'{side}_pt_level2_id',
                                      mesh_index_col=f'{side}_pt_mesh_ind',
                                      inplace=True)
    return syn_df


def get_level2_synapses(root_id,
                        l2dict,
                        client,
                        synapse_table,
                        remove_self=True,
                        pre=True,
                        post=True):

    if pre is True:
        pre_syn_df = _mapped_synapses(
            root_id, client, l2dict, 'pre', synapse_table, remove_self=remove_self)
    else:
        pre_syn_df = None

    if pre is True:
        post_syn_df = _mapped_synapses(
            root_id, client, l2dict, 'post', synapse_table, remove_self=remove_self)
    else:
        post_syn_df = None
    return pre_syn_df, post_syn_df


# def _get_level2_synapses(root_id, l2dict, client, synapse_table):
#     pre_syn_df = client.materialize.query_table(
#         synapse_table, filter_equal_dict={'pre_pt_root_id': root_id})
#     pre_syn_df = pre_syn_df.query(
#         'pre_pt_root_id != post_pt_root_id').reset_index(drop=True)

#     post_syn_df = client.materialize.query_table(
#         synapse_table, filter_equal_dict={'post_pt_root_id': root_id})
#     post_syn_df = post_syn_df.query(
#         'pre_pt_root_id != post_pt_root_id').reset_index(drop=True)

#     pre_lvl2ids = client.chunkedgraph.get_roots(
#         pre_syn_df['pre_pt_supervoxel_id'].values, stop_level=2, timestamp=client.materialize.get_timestamp())
#     post_lvl2ids = client.chunkedgraph.get_roots(
#         post_syn_df['post_pt_supervoxel_id'].values, stop_level=2, timestamp=client.materialize.get_timestamp())

#     pre_syn_df['pre_level2_id'] = pre_lvl2ids
#     post_syn_df['post_level2_id'] = post_lvl2ids

#     pre_syn_df['pre_mind'] = pre_syn_df['pre_level2_id'].apply(
#         lambda x: l2dict[x])
#     post_syn_df['post_mind'] = post_syn_df['post_level2_id'].apply(
#         lambda x: l2dict[x])
#     return pre_syn_df, post_syn_df
