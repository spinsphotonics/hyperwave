"""Utility functions."""

from __future__ import annotations

from jax.typing import ArrayLike

from .typing import Grid, Int3, Subfield


def at(field: ArrayLike, offset: Int3, shape: Int3):
    """Modify ``shape`` values of ``field`` at ``offset``."""
    return field.at[
        ...,
        offset[0] : offset[0] + shape[0],
        offset[1] : offset[1] + shape[1],
        offset[2] : offset[2] + shape[2],
    ]


def get(field: ArrayLike, offset: Int3, shape: Int3):
    """Returns ``shape`` values of ``field`` at ``offset``."""
    return field[
        ...,
        offset[0] : offset[0] + shape[0],
        offset[1] : offset[1] + shape[1],
        offset[2] : offset[2] + shape[2],
    ]


def problem_shape(
    grid: Grid,
    permittivity: ArrayLike,
    conductivity: ArrayLike,
    source_field: Subfield,
) -> Int3:
    """``(xx, yy, zz)`` of problem domain."""
    shape = tuple(du.shape[0] for du in grid.du)

    if not all(du.ndim == 2 and du.shape[1] == 2 for du in grid.du):
        raise ValueError(
            f"grid spacings must be of shape ``[:, 2]`` but got shapes of "
            f"{grid.dx.shape}, {grid.dy.shape}, and {grid.dz.shape} instead."
        )

    if permittivity.shape[-3:] != shape or conductivity.shape[-3:] != shape:
        raise ValueError(
            f"Permittivity and conductivity arrays must match grid shape of "
            f"{shape}, instead got shapes of {permittivity.shape} and "
            f"{conductivity.shape}"
        )

    if any(u < 0 for u in source_field.offset) or any(
        a + b > s
        for a, b, s in zip(source_field.offset, source_field.field.shape[-3:], shape)
    ):
        raise ValueError(
            f"Source must be constrained to within the problem domain, but got "
            f"a source offset of {source_field.offset} and a source field "
            f"shape of {source_field.field.shape} with a problem domain of "
            f"shape {shape}"
        )

    return shape
