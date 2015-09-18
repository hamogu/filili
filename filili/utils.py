



def get_flat_element(nested, name, include_missing=True):
    '''Return a genrator that parses nested lists of lists of dicts.

    The inner most elements have to be dicts. Construct a flat list of
    the values of elements called "name" in those dicts.

    Parameters
    ----------
    nested : list
        An arbitraritly deeply nested list of lists. The inner most elements have
        to be dicts.
    name : string
        Key for the dicts
    include_missing : bool
        If ``True`` return ``None`` for dicts missing the key "name", if ``False``
        skip those dicts.

    Returns
    -------
    gen : generator
    '''
    if isinstance(nested, dict):
        if name in nested:
            yield nested[name]
        else:
            if include_missing:
                yield None
            else:
                return
    else:
        for sublist in nested:
            for element in get_flat_element(sublist, name):
                yield element
