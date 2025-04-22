def del_keys(d, keys):
    for key in keys:
        if key in d:
            del d[key]
    return d


def rename_key(config: dict, old_key: str, new_key: str, default=None):
    if new_key not in config:
        config[new_key] = config.pop(old_key, default)
    return config
