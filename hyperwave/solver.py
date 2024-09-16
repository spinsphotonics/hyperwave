from functools import partial
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import bicgstab
from jax.typing import ArrayLike


class FreqBand(NamedTuple):
    start: float
    end: float
    num: int

    @property
    def values(self):
        return jnp.linspace(self.start, self.end, self.num)[:, None, None, None, None]


class Domain(NamedTuple):
    grid: Tuple[ArrayLike, ArrayLike, ArrayLike]
    freq_band: FreqBand
    permittivity: ArrayLike
    conductivity: ArrayLike

    @property
    def shape(self):
        return (self.freq_band.num,) + self.permittivity.shape

    def operator(self, x, freq_index=None):
        w = self.freq_band.values
        if freq_index is not None:
            w = w[freq_index]

        curl_fwd, curl_bwd = tuple(
            partial(_curl, grid=self.grid, is_forward=f) for f in (True, False)
        )

        return (
            curl_fwd(curl_bwd(x))
            - (w**2) * (self.permittivity - 1j * self.conductivity / w) * x
        )


class Source(NamedTuple):
    values: ArrayLike
    offset: Tuple[int, int, int]

    def full(self, domain: Domain):
        src = -1j * domain.freq_band.values * self.values
        return (
            jnp.zeros(domain.shape, dtype=src.dtype)
            .at[
                :,
                :,
                self.offset[-3] : self.offset[-3] + self.values.shape[-3],
                self.offset[-2] : self.offset[-2] + self.values.shape[-2],
                self.offset[-1] : self.offset[-1] + self.values.shape[-1],
            ]
            .set(src)
        )


def field_solve(
    domain: Domain,
    source: Source,
    init_field: ArrayLike | None = None,
    tol: float = 1e-3,
    max_iters: int | None = None,
) -> jax.Array:

    # def operator(w, x):
    #     return (
    #         curl_fwd(curl_bwd(x))
    #         - (w**2) * (domain.permittivity - 1j * domain.conductivity / w) * x
    #     )

    # # Expand source across the whole domain.
    # src = -1j * domain.freq_band.values * source.values
    # src = (
    #     jnp.zeros(domain.shape, dtype=src.dtype)
    #     .at[
    #         :,
    #         :,
    #         source.offset[-2] : source.offset[-2] + source.values.shape[-2],
    #         source.offset[-1] : source.offset[-1] + source.values.shape[-1],
    #         source.offset[-0] : source.offset[-0] + source.values.shape[-0],
    #     ]
    #     .set(src)
    # )

    if init_field is None:
        init_field = jnp.zeros_like(source.full(domain))

    def solve(op, b, x0):
        field, _ = bicgstab(A=op, b=b, x0=x0, tol=tol, maxiter=max_iters)
        return field

    return jnp.stack(
        [
            solve(
                partial(domain.operator, freq_index=i),
                source.full(domain)[i],
                init_field[i],
            )
            for i in range(domain.freq_band.num)
        ]
    )


def _curl(
    field: ArrayLike, grid: Tuple[ArrayLike, ArrayLike, ArrayLike], is_forward: bool
) -> jax.Array:
    """Curl of ``field`` on ``grid`` with ``is_forward=False`` for E-field."""

    fx, fy, fz = [field[..., i, :, :, :] for i in range(3)]
    dx, dy, dz = [
        partial(
            _spatial_diff,
            delta=grid[axis],
            axis=axis,
            is_forward=is_forward,
        )
        for axis in range(-3, 0)
    ]
    return jnp.stack([dy(fz) - dz(fy), dz(fx) - dx(fz), dx(fy) - dy(fx)], axis=-4)


def _spatial_diff(
    field: ArrayLike,
    delta: ArrayLike,
    axis: int,
    is_forward: bool,
) -> jax.Array:
    """Spatial differences of ``field`` along ``axis``."""
    # Make the grid spacing align with the difference axis.
    delta = jnp.expand_dims(delta, range(axis, 0))

    # Take the forward- or backward-difference which either reach ahead or
    # behind, respectively, one grid cell.
    if is_forward:
        return (jnp.roll(field, shift=-1, axis=axis) - field) / delta[axis]
    return (field - jnp.roll(field, shift=+1, axis=axis)) / delta[axis]
