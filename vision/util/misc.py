"""
misc utils
"""


def safe_layername(layer):
    if isinstance(layer, list):
        return "-".join(map(str, layer))
    else:
        return layer
