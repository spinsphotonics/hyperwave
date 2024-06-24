"""Basic types."""

from __future__ import annotations

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

# NOTE: Please avoid including logic here! Included types should be trivially simple.

# Tuple of 3 integers, used for ``(x, y, z)`` data.
Int3 = Tuple[int, int, int]


class Band(NamedTuple):
    """Describes ``num`` regularly spaced values within ``[start, stop].``

    The suggested convention for ``num=1`` is that the :py:class:`Band`
    represents the single-element array with value ``(start + stop) / 2``.

    Args:
        start: Extremal value of the band.
        stop: Other extremal value of the band.
        num: Number of equally-spaced values within ``[start, stop]``.

    """

    start: float
    stop: float
    num: int

    @property
    def values(self) -> jax.Array:
        """Values represented by ``band``."""
        if self.num == 1:
            return jnp.array([(self.start + self.stop) / 2])
        else:
            return jnp.linspace(self.start, self.stop, self.num)


class Grid(NamedTuple):
    """Defines the Yee-lattice for the simulation volume.

    The Yee-lattice is a Cartesian grid of Yee cells which must be defined
    along each spatial axis by the cell-to-cell spacings between cell centers
    as well as between cell boundaries, because of the offset nature of field
    components in the Yee cell.

    Each axis is thus defined by a ``(uu, 2)`` array of spacing values where

    * ``[:, 0]`` values are the center-to-center spacings, and
    * ``[:, 1]`` values are the boundary-to-boundary spacings,

    and where the ``[:, 0]`` intervals are shifted in the negative direction
    relative to the ``[:, 1]`` intervals.

    The convention for the absolute location of field components is thus
    suggested to be as follows

    * place the boundary of the first cell of the Yee lattice at the ``0``
      axis position,
    * place the offset components of the first cell at position equal to
      ``dxyz[0, 0] / 2`` where ``dxyz`` stands in for the spacing array along
      the relevant axis,
    * then let the ``i`` th cell boundary be at position ``sum(dxyz[:i, 1])`` and
      the ``i`` th cell center be at position
      ``sum(dxyz[1:i, 0]) + dxyz[0, 0] / 2``.


    Args:
        dx: ``(xx, 2)`` of Yee-lattice spacings along the x-axis.
        dy: ``(yy, 2)`` of Yee-lattice spacings along the y-axis.
        dz: ``(zz, 2)`` of Yee-lattice spacings along the z-axis.

    """

    dx: ArrayLike
    dy: ArrayLike
    dz: ArrayLike

    @property
    def du(self):
        return (self.dx, self.dy, self.dz)


class Subfield(NamedTuple):
    """Field defined at ``offset`` in space.

    Args:
        offset: ``(x0, y0, z0)`` cell in the simulation grid at which
          ``field`` is defined.
        field: ``(3, xx0, yy0, zz0)`` array defining the field at ``offset``.

    """

    offset: Int3
    field: ArrayLike


class Range(NamedTuple):
    """Describes values ``start + i * interval`` for ``i`` in ``[0, num)``.

    Args:
        start: Value at which the the range starts.
        interval: Spacing between values.
        num: Total number of values in the range.

    """

    start: int
    interval: int
    num: int


class Volume(NamedTuple):
    """Identifies a volume of size ``shape`` at ``offset`` in space.

    Args:
        offset: ``(x0, y0, z0)`` point at which the volume is offset.
        shape: ``(xx0, yy0, zz0)`` shape of the volume.


    """

    offset: Int3
    shape: Int3
