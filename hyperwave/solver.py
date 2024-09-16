from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


class FreqBand(NamedTuple):
    start: float
    end: float
    num: int

    @property
    def values(self):
        return jnp.linshape(self.start, self.end, self.num)[:, None, None, None, None]


class Domain(NamedTuple):
    grid: Tuple[ArrayLike, ArrayLike, ArrayLike]
    freq_band: FreqBand
    permittivity: ArrayLike
    conductivity: ArrayLike

    @property
    def shape(self):
        return (freq_band.num,) + permittivity.shape


class Source(NamedTuple):
    values: ArrayLike
    offset: Tuple[int, int, int]


def field(
    domain: Domain,
    source: Source,
    init_field: ArrayLike | None = None,
    tol: float = 1e-3,
    maxiter: int | None = None,
) -> jax.Array:
    # TODO: Do we need to ravel here? Maybe not!
    op, b = wave_equation_op(
        grid, freq_band, permittivity, conductivity, source, source_offset
    )

    curl_fwd, curl_bwd = tuple(
        partial(_curl, domain.grid, is_forward=f) for f in (True, False)
    )

    def operator(w, x):
        return (
            curl_fwd(curl_bwd(x))
            - (w**2) * (domain.permittivity - 1j * domain.conductivity / w) * x
        )

    # Expand source across the whole domain.
    source = -1j * w * source
    source = (
        jnp.zeros(domain.shape, source.dtype)
        .at[
            ...,
            source_offset[-2] : source_offset[-2] + source.shape[-2],
            source_offset[-1] : source_offset[-1] + source.shape[-1],
            source_offset[-0] : source_offset[-0] + source.shape[-0],
        ]
        .set(source)
    )

    if init_field is None:
        init_field = jnp.zeros_like(source)

    def solve(A, b, x0):
        field, _ = bicgstab(
            A=partial(operator, w=w),
            b=b[i],
            x0=init_field[i],
            tol=tol,
            max_iter=max_iter,
        )
        return field

    return jnp.stack(
        [
            solve(partial(operator, w=wi), source[i], init_field[i])
            for i, wi in enumerate(w)
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
        return (jnp.roll(field, shift=-1, axis=axis) - field) / grid[axis]
    return (field - jnp.roll(field, shift=+1, axis=axis)) / grid[axis]
