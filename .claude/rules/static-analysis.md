---
description: Static analysis standards
---

# Static Analysis Standards

**All code including examples must pass the static analysis.**
The `examples/` folder is NOT excluded from type checking or linting.

## Type Annotations Required

All code must have proper type annotations:

```python
def visualize_trees(
    original_clf: DecisionTreeClassifier,
    simplified_clf: DecisionTreeClassifier,
    feature_names: list[str],
    class_names: list[str],
    save_path: str | None = None,
) -> None:
    ...
```

## Configuration

The static analysis configuration for all tools is in the `pyproject.toml` file.

## Tools

* [ruff](https://github.com/astral-sh/ruff)
* [mypy](https://github.com/python/mypy)
* [pyright](https://github.com/microsoft/pyright)
