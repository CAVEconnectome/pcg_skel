import datetime
from caveclient.frameworkclient import CAVEclientFull
import warnings

try:
    from annotationframeworkclient.frameworkclient import FrameworkClientFull

    check_afc = True
except:
    FrameworkClientFull = type(None)
    check_afc = False


def annotation_to_level2_id(
    df,
    client,
    bound_pt_columns="pt",
    l2_suffix="_level2_id",
    sv_columns=None,
    l2_columns=None,
    inplace=False,
    timestamp=None,
):
    """Add or more level2_id columns to a dataframe based on supervoxel columns

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with one or more supervoxel columns
    client : CAVEclient or ChunkedgraphClient
        Client for interacting with the chunkedgraph
    bound_pt_columns : str or list-like, optional
        List of bound spatial point names, prefix only. E.g. 'pt' for an annotation with
        'pt_position', 'pt_supervoxel_id', and 'pt_root_id'. Optional, by default 'pt'.
    l2_suffix : str, optional
        Suffix to use for new level 2 id column, by default '_level2_id'
    sv_columns : str or list-like, optional
        Explicit list of level 2 columns, not needed if using bound_pt_columns. By default None
    l2_columns : [type], optional
        Explicit list of desired level 2 columns, not needed if using bound_pt_columns. By default None
    inplace : bool, optional
        If True, change the dataframe in place, by default False
    timestamp : datetime or None, optional
        Timestamp to lookup the mapping. If None, uses the materialization version timestamp.
        Default is None.

    Returns
    -------
    [type]
        [description]
    """
    if isinstance(bound_pt_columns, str):
        bound_pt_columns = [bound_pt_columns]
    if sv_columns is None:
        sv_columns = [f"{c}_supervoxel_id" for c in bound_pt_columns]
        if l2_columns is None:
            l2_columns = [f"{c}{l2_suffix}" for c in bound_pt_columns]
    elif l2_columns is None:
        l2_columns = [f"{c}{l2_suffix}" for c in sv_columns]

    if timestamp is None:
        timestamp = client.materialize.get_timestamp()

    if isinstance(client, CAVEclientFull):
        pcg_client = client.chunkedgraph
    elif check_afc and isinstance(client, FrameworkClientFull):
        warnings.warn(
            DeprecationWarning(
                "annotationframeworkclient is depricated. Please switch to using caveclient."
            )
        )
        pcg_client = client.chunkedgraph
    else:
        pcg_client = client

    if not inplace:
        df = df.copy()

    for col, l2_col in zip(sv_columns, l2_columns):
        level2_ids = pcg_client.get_roots(
            df[col].values,
            stop_layer=2,
            timestamp=timestamp,
        )
        df[l2_col] = level2_ids
    return df


def annotation_to_mesh_index(
    df,
    l2dict,
    level2_id_col="pt_level2_id",
    mesh_index_col="pt_mesh_ind",
    inplace=False,
):
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


def _mapped_synapses(
    root_id,
    client,
    l2dict,
    side,
    synapse_table,
    remove_self,
    live_query,
    timestamp,
):
    if live_query:
        syn_df = client.materialize.live_query(
            synapse_table,
            filter_equal_dict={f"{side}_pt_root_id": root_id},
            timestamp=timestamp,
        )
    else:
        syn_df = client.materialize.query_table(
            synapse_table,
            filter_equal_dict={f"{side}_pt_root_id": root_id},
        )

    if remove_self:
        syn_df = syn_df.query("pre_pt_root_id != post_pt_root_id").reset_index(
            drop=True
        )
    syn_df = annotation_to_level2_id(
        syn_df,
        client,
        bound_pt_columns=f"{side}_pt",
        inplace=True,
        timestamp=timestamp,
    )
    syn_df = annotation_to_mesh_index(
        syn_df,
        l2dict,
        level2_id_col=f"{side}_pt_level2_id",
        mesh_index_col=f"{side}_pt_mesh_ind",
        inplace=True,
    )
    return syn_df


def get_level2_synapses(
    root_id,
    l2dict,
    client,
    synapse_table,
    remove_self=True,
    pre=True,
    post=True,
    live_query=False,
    timestamp=None,
):
    live_query = timestamp is not None

    if timestamp is None:
        timestamp = datetime.datetime.utcnow()
    if pre is True:
        pre_syn_df = _mapped_synapses(
            root_id,
            client,
            l2dict,
            "pre",
            synapse_table,
            remove_self=remove_self,
            live_query=live_query,
            timestamp=timestamp,
        )
    else:
        pre_syn_df = None

    if post is True:
        post_syn_df = _mapped_synapses(
            root_id,
            client,
            l2dict,
            "post",
            synapse_table,
            remove_self=remove_self,
            live_query=live_query,
            timestamp=timestamp,
        )
    else:
        post_syn_df = None

    return pre_syn_df, post_syn_df
