def wrap_list(val):
    if isinstance(val, list):
        return val
    else:
        return [val]
