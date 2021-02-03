from sqlitedict import SqliteDict
import orjson
import numpy as np


def lookup_cached_ids(l2_ids, cache_file):

    l2means = []
    is_cached = []

    with SqliteDict(cache_file, decode=orjson.loads, flag='c') as cache_dict:
        for l2id in l2_ids:
            loc = cache_dict.get(str(l2id), None)
            if loc is not None:
                l2means.append(loc)
                is_cached.append(True)
            else:
                is_cached.append(False)

    if len(l2means) > 0:
        l2means = np.vstack(l2means)
    else:
        l2means = np.zeros((0, 3), dtype=float)

    return l2means, np.array(is_cached)


def save_ids_to_cache(l2_ids, l2_locs, cache_file):
    """Save computed locations back to the l2 cache.

    Parameters
    ----------
    l2_ids : list-like
        List of N layer 2 ids
    l2_locs : np.array
        Nx3 array of locations
    cache_file : str

    """
    ii = 0
    with SqliteDict(cache_file, encode=orjson.dumps, flag='c') as cache_dict:
        for k, v in zip(l2_ids, l2_locs):
            if not np.any(np.isnan(v)):
                cache_dict[str(k)] = v.tolist()
        cache_dict.commit()
    pass
