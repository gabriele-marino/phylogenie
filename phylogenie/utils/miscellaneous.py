from typing import Any


def flatten_dict(
    d: dict[Any, Any],
    sep: str = ".",
    *,
    parent_key: str = "",
) -> dict[str, Any]:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, sep, parent_key=new_key))
        else:
            items[new_key] = v
    return items
