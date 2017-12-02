"""
Manual garbage collection: reference counting
"""


def _get_ref_pointers(obs_pointers):
    for pt in obs_pointers:
        assert pt.startswith('obs:')
    return ['ref-'+pt for pt in obs_pointers]


def incr_ref_count(client, obs_pointers):
    ref_pointers = _get_ref_pointers(obs_pointers)
    client.mincr(ref_pointers)


def decr_ref_count(client, obs_pointers, delete=True):
    """
    Delete obs when its ref count drops down to 0

    Args:
        delete: if True, execute delete on the Redis server

    Returns:
        evicted obs_pointers
    """
    ref_pointers = _get_ref_pointers(obs_pointers)
    ref_counts = client.mdecr(ref_pointers)
    # only evict when ref count drops to 0
    evict_obs_pointers = [obs_pointers[i]
                          for i in range(len(obs_pointers))
                          if ref_counts[i] <= 0]
    if delete:
        # mass delete exp and obs (only when ref drop to 0) on Redis
        client.mdel(evict_obs_pointers)
    return evict_obs_pointers
