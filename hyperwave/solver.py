from functools import partial
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax.experimental.sparse.linalg import lobpcg_standard
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

    @property
    def shape(self):
        return self.values.shape[-3:]

    def full_term(self, domain):
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

    @staticmethod
    def from_mode(mode: ArrayLike, offset: Int3, axis: int):
        values = (jnp.zeros_like(mode[:, 0]), mode[:, 0], mode[:, 1])
        return Source(
            values=jnp.stack([values[(3 - axis + i) % 3] for i in range(3)], axis=1),
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

    # TODO: This is tricky! leading dimension for ``x`` is sometimes used as mode dimension, sometimes used as frequency dimension.
    def mode_operator(
        self, x: ArrayLike, offset: Int3, shape: Int3, axis: int, freq_index=None
    ):
        subdomain = self._subdomain(offset, shape)

        dfi, dbi, dfj, dbj = [
            partial(
                subdomain._spatial_diff,
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

        # TODO: If the domain is singular in one direction, can we do this more simply?
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

        w = self.freq_band.values
        if freq_index is not None:
            w = w[freq_index]
        ei, ej, ek = tuple(subdomain.permittivity[(i + 1) % 3] for i in range(3))
        eji = jnp.stack([ej, ei], axis=0)
        return w**2 * eji * x + eji * curl_to_ij(curl_to_k(x) / ek) + grad(div(x))

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

    def _subdomain(self, offset: Int3, shape: Int3):
        return Domain(
            grid=tuple(g[u0 : u0 + uu] for g, u0, uu in zip(self.grid, offset, shape)),
            freq_band=self.freq_band,
            permittivity=Domain._subvolume(self.permittivity, offset, shape),
            conductivity=Domain._subvolume(self.conductivity, offset, shape),
        )

    @staticmethod
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
        init_field = jnp.zeros_like(source.full_term(domain))

    def solve(op, b, x0):
        field, _ = bicgstab(A=op, b=b, x0=x0, tol=tol, maxiter=max_iters)
        return field

    return jnp.stack(
        [
            jax.lax.custom_linear_solve(
                matvec=partial(domain.wave_operator, freq_index=i),
                b=source.full_term(domain)[i],
                solve=partial(solve, x0=init_field[i]),
                symmetric=True,
            )
            for i in range(domain.freq_band.num)
        ]
    )


def field_error(domain: Domain, source: Source, field: ArrayLike):
    src = source.full_term(domain)
    return _norm_by_freq(domain.wave_operator(field) - src) / _norm_by_freq(src)


def mode_solve(
    domain: Domain,
    offset: Int3,
    shape: Int3,
    mode_num: int,
    axis: int,
    random_seed: int = 0,
) -> Tuple[jax.Array, jax.Array]:
    # TODO: Need to document that we are using the frequency axis as the mode axis.
    field_shape = (mode_num + 1, 2) + shape

    def _to_field(x):
        return jnp.reshape(x.T, (-1, 2) + shape)

    def _flatten(field):
        return jnp.reshape(field, (field.shape[0], -1)).T

    def op(x, freq_index):
        return _flatten(
            domain.mode_operator(
                _to_field(x),
                offset=offset,
                shape=shape,
                axis=axis,
                freq_index=freq_index,
            )
        )

    # TODO: Not sure about this...
    x0 = jax.random.normal(jax.random.PRNGKey(random_seed), field_shape)

    # TODO: Try to vmap this?
    wavevectors, fields = [], []
    for i in range(domain.freq_band.num):
        betas_squared, x, _ = lobpcg_standard(partial(op, freq_index=i), _flatten(x0))
        fields.append(_to_field(x)[-1])
        wavevectors.append(jnp.sqrt(betas_squared)[-1])

    return Source.from_mode(jnp.stack(fields), offset, axis)


def mode_wavevector(domain: Domain, source: Source, axis: int):
    x = source.to_mode(axis)
    y = domain.mode_operator(x, source.offset, source.shape, axis)
    return jnp.sqrt(
        _norm_by_freq(x * y) / _norm_by_freq(x * x)
    )  # TODO: Why is sqrt needed?


def mode_error(domain: Domain, source: Source, axis: int):
    x = source.to_mode(axis)
    lx = jnp.square(mode_wavevector(domain, source, axis)) * x
    return _norm_by_freq(
        domain.mode_operator(x, source.offset, source.shape, axis) - lx
    ) / _norm_by_freq(x)


def cut_solve(domain: Domain, field: ArrayLike, axis: int, pos: int, is_forward: bool):
    field = cut_field(field, axis, pos, is_forward)
    a, b = [0, 0, 0], list(domain.shape[-3:])
    a[axis] = pos
    b[axis] = pos + 2
    x = domain.wave_operator(field)[..., a[0] : b[0], a[1] : b[1], a[2] : b[2]] / (
        -1j * domain.freq_band.values
    )
    return Source(offset=a, values=x)


def cut_field(field: ArrayLike, axis: int, pos: int, is_forward: bool):
    a, b = [0, 0, 0], list(field.shape[-3:])
    if is_forward:
        b[axis] = pos + 1
    else:
        a[axis] = pos + 1
    return field.at[..., a[0] : b[0], a[1] : b[1], a[2] : b[2]].set(0.0)


def _norm_by_freq(x: ArrayLike):
    return jnp.linalg.norm(jnp.reshape(x, (x.shape[0], -1)), axis=-1)
