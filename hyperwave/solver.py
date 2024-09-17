from functools import partial
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import bicgstab
from jax.typing import ArrayLike

Int3 = Tuple[int, int, int]


class FreqBand(NamedTuple):
    start: float
    end: float
    num: int

    @property
    def values(self):
        return jnp.linspace(self.start, self.end, self.num)[:, None, None, None, None]


class Source(NamedTuple):
    values: ArrayLike
    offset: Int3

    def full(self, domain):
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


class Domain(NamedTuple):
    grid: Tuple[ArrayLike, ArrayLike, ArrayLike]
    freq_band: FreqBand
    permittivity: ArrayLike
    conductivity: ArrayLike

    @property
    def shape(self):
        return (self.freq_band.num,) + self.permittivity.shape

    def wave_operator(self, x: ArrayLike, freq_index=None):
        w = self.freq_band.values
        if freq_index is not None:
            w = w[freq_index]

        curl_fwd, curl_bwd = tuple(
            partial(self._curl, is_forward=f) for f in (True, False)
        )

        return (
            curl_fwd(curl_bwd(x))
            - (w**2) * (self.permittivity - 1j * self.conductivity / w) * x
        )

    def _curl(self, x: ArrayLike, is_forward: bool) -> jax.Array:
        fx, fy, fz = [x[..., i, :, :, :] for i in range(3)]
        dx, dy, dz = [
            partial(self._spatial_diff, axis=axis, is_forward=is_forward)
            for axis in range(-3, 0)
        ]
        return jnp.stack([dy(fz) - dz(fy), dz(fx) - dx(fz), dx(fy) - dy(fx)], axis=-4)

    def _spatial_diff(self, x: ArrayLike, axis: int, is_forward: bool) -> jax.Array:
        """Spatial differences of ``field`` along ``axis``."""
        # Make the grid spacing align with the difference axis.
        delta = jnp.expand_dims(self.grid[axis], range(axis, 0))

        # Take the forward- or backward-difference which either reach ahead or
        # behind, respectively, one grid cell.
        if is_forward:
            return (jnp.roll(x, shift=-1, axis=axis) - x) / delta[axis]
        return (x - jnp.roll(x, shift=+1, axis=axis)) / delta[axis]

    def mode_operator(self, x, axis: int):
        subdomain = self._subdomain(x.offset, x.values.shape)
        axis_inds = tuple((axis + d) % 3 for d in range(1, 3))
        mode_field = x.values[:, axis_inds, ...]
        mode_field = subdomain._mode_operator_impl(mode_field, axis)
        values = (
            jnp.zeros_like(x.values)
            .at[:, axis_inds + (axis,), ...]
            .set(jnp.concat([mode_fields, jnp.zeros_like(mode_fields)[:, 0]]))
        )
        return Source(values=values, offset=x.offset)

    # def mode_operator(self, x: ArrayLike, offset: Int3, shape: Int3, axis: int):
    #     return self._subdomain(offset, shape)._mode_operator_impl(x, axis)

    def _subdomain(self, offset: Int3, shape: Int3):
        return Domain(
            grid=tuple(g[u0 : u0 + uu] for g, u0, uu in (self.grid, offset, shape)),
            freq_band=self.freq_band,
            permittivity=self._subvolume(self.permittivity, offset, shape),
            conductivity=self._subvolume(self.conductivity, offset, shape),
        )

    @classmethod
    def _subvolume(u: ArrayLike, offset: Int3, shape: Int3) -> ArrayLike:
        return u[
            ...,
            offset[-3] : offset[-3] + shape[-3],
            offset[-2] : offset[-2] + shape[-2],
            offset[-1] : offset[-1] + shape[-1],
        ]

    def _mode_operator_impl(self, x: ArrayLike, axis: int):
        dfi, dbi, dfj, dbj = [
            partial(
                _spatial_diff,
                delta=self.grid[(axis + axis_shift) % 3],
                axis=((axis + axis_shift) % 3) - 3,
                is_forward=is_forward,
            )
            for (axis_shift, is_forward) in (
                (1, True),
                (1, False),
                (2, True),
                (2, False),
            )
        ]

        def _split(u):
            return jnp.split(u, indices_or_sections=2, axis=1)

        def _concat(u):
            return jnp.concatenate(u, axis=1)

        def curl_to_k(u):
            ui, uj = _split(u)
            return dbi(uj) - dbj(ui)

        def curl_to_ij(u):
            return _concat([-dfj(u), dfi(u)])

        def div(u):
            ui, uj = _split(u)
            return dfi(ui) + dfj(uj)

        def grad(u):
            return _concat([dbi(u), dbj(u)])

        omega = (freq_band.start + freq_band.stop) / 2
        ei, ej, ek = tuple(self.permittivity[(i + 1) % 3] for i in range(3))
        eji = jnp.stack([ej, ei], axis=0)
        return omega**2 * eji * x + eji * curl_to_ij(curl_to_k(x) / ek) + grad(div(x))


def field_solve(
    domain: Domain,
    source: Source,
    init_field: ArrayLike | None = None,
    tol: float = 1e-3,
    max_iters: int | None = None,
) -> jax.Array:

    if init_field is None:
        init_field = jnp.zeros_like(source.full(domain))

    def solve(op, b, x0):
        field, _ = bicgstab(A=op, b=b, x0=x0, tol=tol, maxiter=max_iters)
        return field

    return jnp.stack(
        [
            solve(
                partial(domain.wave_operator, freq_index=i),
                source.full(domain)[i],
                init_field[i],
            )
            for i in range(domain.freq_band.num)
        ]
    )


# TODO: need to figure out stuff?
def mode_solve(
    domain: Domain, offset: Int3, shape: Int3, mode_num: int, axis: int
) -> jax.Array:
    # field_shape = tuple(d.shape[0] for d in grid.du)
    # shape = (2 * prod(field_shape), num_modes)
    field_shape = (2,) + shape + (mode_num + 1,)

    # TODO: The problem here is that we need to remove the longitudinal axis.
    # For this reason we should just use a separate "mode field" and have
    # some way to convert this to a Source object.
    def op(x):
        src = Source(values=jnp.reshape(x, field_shape), offset=offset)
        src = domain.mode_operator(src, axis)
        return jnp.reshape(src.values, (-1, field_shape[-1]))

    # TODO: Not sure about this...
    x0 = jax.random.normal(jax.random.PRNGKey(random_seed), field_shape)

    # betas_squared, x, _ = lobpcg_standard(op, jnp.reshape(x0, (-1, field_shape[-1])))

    return jnp.reshape(x, field_shape)[..., -1]
    # errs = jnp.linalg.norm(op(x) - betas_squared * u, axis=0)
    modes = jnp.reshape(u.T, (-1, 2) + field_shape)
    betas = jnp.sqrt(betas_squared)
    return modes, betas, errs
