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

    def full_values(self, domain):
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

    @classmethod
    def from_mode(mode: ArrayLike, offset: Int3, axis: int):
        values = (jnp.zeros_like(mode[:, 0]), mode[:, 0], mode[:, 1])
        return Source(
            values=jnp.stack([values[(axis + i) % 3] for i in range(3)], axis=1),
            offset=offset,
        )

    def to_mode(self, axis: int):
        return self.values[:, tuple((axis + i) % 3 for i in range(1, 3))]


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

    def mode_operator(self, x: ArrayLike, offset: Int3, shape: Int3, axis: int):
        subdomain = self._subdomain(offset, shape)

        dfi, dbi, dfj, dbj = [
            partial(
                _spatial_diff,
                delta=subdomain.grid[(axis + axis_shift) % 3],
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
        ei, ej, ek = tuple(subdomain.permittivity[(i + 1) % 3] for i in range(3))
        eji = jnp.stack([ej, ei], axis=0)
        return omega**2 * eji * x + eji * curl_to_ij(curl_to_k(x) / ek) + grad(div(x))

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


def field_solve(
    domain: Domain,
    source: Source,
    init_field: ArrayLike | None = None,
    tol: float = 1e-3,
    max_iters: int | None = None,
) -> jax.Array:

    if init_field is None:
        init_field = jnp.zeros_like(source.full_values(domain))

    def solve(op, b, x0):
        field, _ = bicgstab(A=op, b=b, x0=x0, tol=tol, maxiter=max_iters)
        return field

    return jnp.stack(
        [
            solve(
                partial(domain.wave_operator, freq_index=i),
                source.full_values(domain)[i],
                init_field[i],
            )
            for i in range(domain.freq_band.num)
        ]
    )


def field_error(domain: Domain, source: Source, field: ArrayLike):
    src = source.full_values(domain)
    return _norm_by_freq(domain.wave_operator(field) - src) / _norm_by_freq(src)


def mode_solve(
    domain: Domain, offset: Int3, shape: Int3, mode_num: int, axis: int
) -> Tuple[jax.Array, jax.Array]:
    # field_shape = tuple(d.shape[0] for d in grid.du)
    # shape = (2 * prod(field_shape), num_modes)
    field_shape = (2,) + shape + (mode_num + 1,)

    def op(x):
        return jnp.reshape(
            domain.mode_operator(jnp.reshape(x, field_shape), axis),
            (-1, field_shape[-1]),
        )

    # TODO: Not sure about this...
    x0 = jax.random.normal(jax.random.PRNGKey(random_seed), field_shape)

    # TODO: Need to modify so that we solve for indiv. frequencies.
    betas_squared, x, _ = lobpcg_standard(op, jnp.reshape(x0, (-1, field_shape[-1])))
    return (
        Source.from_mode(jnp.reshape(x, field_shape)[..., -1], offset, axis),
        jnp.sqrt(betas_squared)[..., -1],
    )


def mode_error(domain: Domain, source: Source, wavevectors: ArrayLike, axis: int):
    lx = jnp.square(wavevectors) * x
    return _norm_by_freq(domain.mode_operator(x, axis) - lx) / _norm_by_freq(x)


def _norm_by_freq(x: ArrayLike):
    return jnp.linalg.norm(jnp.reshape(x, (x.shape[0], -1)), axis=-1)
