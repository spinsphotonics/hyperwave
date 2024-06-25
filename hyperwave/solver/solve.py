"""Solves the wave equation via FDTD simulation."""

from __future__ import annotations

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from . import fdtd, grids, sampling, utils
from .typing import Band, Grid, Int3, Range, Subfield, Volume


def solve(
    grid: Grid,
    freq_band: Band,
    permittivity: ArrayLike,
    conductivity: ArrayLike,
    source: Subfield,
    err_thresh: float | None,
    max_steps: int,
    output_volumes: Sequence[Volume] | None = None,
) -> Tuple[jax.Array | Tuple[jax.Array, ...], jax.Array | None, int]:
    r"""Solve the time-harmonic electromagnetic wave equation.

    :py:func:`solve` attempts to solve
    :math:`\nabla \times \nabla \times E - \omega^2 \epsilon E = i \omega J`
    where

    * :math:`\nabla \times` is the `curl operation <https://en.wikipedia.org/wiki/Curl_(mathematics)>`_,
    * :math:`E` is the electric-field,
    * :math:`\omega` is the angular frequency,
    * :math:`\epsilon` is the relative permittivity, and
    * :math:`J` is the electric current excitation.

    Dimensionsless units are used such that

    * :math:`\epsilon_0 = 1` and :math:`\mu_0 = 1`, the permittivity and
      permeability of vacuum are set to (dimensionless) :math:`1`,
    * :math:`c = 1`, the speed of light is also equal (dimensionless) :math:`1`,
    * space is also dimensionless, and
    * :math:`\omega = 2 \pi / \lambda_0`, the angular frequency can be defined
      relative to (dimensionless) vacuum wavelength.

    :math:`E`, :math:`\epsilon`, and :math:`J` values are located on the
    `Yee lattice <https://en.wikipedia.org/wiki/Finite-difference_time-domain_method>`_,
    at locations corresponding to the electric field position.

    Supports simple dielectric materials (no disperson or nonlinearity, loss
    supported) via :math:`\epsilon = \epsilon' + i \sigma / \omega`, where
    :math:`\epsilon'` is the real-valued permittivity and :math:`\sigma` is the
    real-valued conductivity. Anisotropy is explicitly supported by virtue that
    the components of :math:`\epsilon` can be arbitrarily valued, although
    off-diagonal values of the permittivity tensor are not supported.

    :py:func:`solve` can obtain solutions to the wave equation for multiple
    regularly-spaced angular frequencies simultaneously; however, each frequency
    is not allowed to have its own independent input current source.

    The error in the wave equation is defined as
    :math:`(1 / \sqrt{n}) \cdot \lVert \nabla \times \nabla \times E - \omega^2 \epsilon E - i \omega J \rVert / \lVert \omega J \rVert`
    where :math:`n` is the number of elements in the solution field :math:`E`.

    For very large simulations (possibly over many frequencies), we may not
    want to store the entirety of the solution fields. In these cases, the
    error computation can be elided and we can choose to return only the desired
    subdomains of the output fields.

    Args:
        grid: Spacing of the simulation grid.
        freq_band: Angular frequencies at which to produce solution fields.
        permittivity: ``(3, xx, yy, zz)`` array of real-valued permittivity
          values.
        conductivity: ``(3, xx, yy, zz)`` array of real-valued conductivity
          values.
        source: Complex-valued current excitation.
        err_thresh: Terminate when the error of the fields at each frequency
          are below this value. If ``None``, do not compute error values and
          instead terminate on the ``max_steps`` condition only. For
          ``err_thresh <= 0``, termination on ``max_steps`` is guaranteed
          without omitting the error computation.
        max_steps: Maximum number of simulation update steps to execute.
          This is a "soft" termination barrier as additional steps beyond
          ``max_steps`` may be needed to extract solution fields.
        output_volumes: If ``None`` (default), then the solution fields are
          returned in their entirety; otherwise, only sub-volumes of the
          solution field are returned.

    Returns:
        ``(outs, errs, num_steps)`` where

        * when ``output_volumes=None``, ``outs`` is a ``(ww, 3, xx, yy, zz)``
          array (where ``ww`` is the number of frequencies requested via
          ``freq_band.num``).
        * when ``output_volumes`` is a n-tuple of :py:class:`Volume` objects,
          ``outs`` is an n-tuple of ``(ww, 3, xxi, yyi, zzi)`` corresponding to
          the ``shape`` parameters in ``output_volumes``.
        * ``errs`` is a ``(ww,)`` array at each frequency, or else ``None`` for
          the case where ``err_thresh=None``.
        * ``num_steps`` is the number of time-domain updates executed.

    """
    shape = utils.problem_shape(grid, permittivity, conductivity, source)
    dt, sample_every_n = sampling_strategy(freq_band, permittivity)
    steps_per_sim = simulation_steps(freq_band, sample_every_n, shape)

    # # TODO: Remove.
    # # Phase stuff.
    # phases = -1 * jnp.pi * jnp.arange(freq_band.num)
    # t = jnp.arange(2 * max_steps) * dt  # TODO: Fix
    # phi = freq_band.values[:, None] * t + phases[:, None]
    # waveform = jnp.sum(jnp.exp(1j * phi), axis=0)

    # Initial stuff.
    state = fdtd.State(  # TODO: Reconcile with fdtd.simulate() way of doing init state.
        step=-1, e_field=jnp.zeros((3,) + shape), h_field=jnp.zeros((3,) + shape)
    )
    if output_volumes is None:
        output_volumes = [Volume(offset=(0, 0, 0), shape=shape)]

    snapshot_range = Range(
        start=-1,
        interval=sample_every_n,
        num=2 * freq_band.num,
    )

    def cond_fn(foo):
        start_step, _, _, errs = foo
        return jnp.logical_and(jnp.max(errs) > err_thresh, start_step < max_steps)

    def body_fn(foo):
        start_step, state, _, _ = foo
        state = fdtd.State(step=-1, e_field=state.e_field, h_field=state.h_field)

        phases = -1 * jnp.pi * jnp.arange(freq_band.num)
        t = (start_step + 1 + jnp.arange(steps_per_sim)) * dt  # TODO: Fix
        phi = freq_band.values[:, None] * t + phases[:, None]
        wvfrm = jnp.sum(jnp.exp(1j * phi), axis=0)

        # Run simulation.
        state, outs = fdtd.simulate(
            dt=dt,
            grid=grid,
            permittivity=permittivity,
            conductivity=conductivity,
            source_field=source,
            source_waveform=wvfrm,  # waveform,
            output_volumes=output_volumes,
            snapshot_range=snapshot_range,
            state=state,
        )

        # Infer time-harmonic fields.
        # TODO: Generalize beyond 1st output.
        t = dt * (
            0.5 + start_step + snapshot_range.interval * jnp.arange(snapshot_range.num)
        )
        freq_fields = sampling.project(outs[0], freq_band, t)

        # Undo phase changes
        freq_fields *= jnp.expand_dims(
            jnp.exp(-1j * phases), axis=range(1, freq_fields.ndim)
        )

        # Compute error.
        errs = wave_equation_error(
            fields=freq_fields,
            freq_band=freq_band,
            permittivity=permittivity,
            conductivity=conductivity,
            source=source,
            grid=grid,
        )

        start_step += steps_per_sim

        return start_step, state, freq_fields, errs
        # print(f"{errs}, {start_step}, {state.step}, {snapshot_range}")

    init_foo = (
        -1,
        state,
        jnp.zeros((freq_band.num, 3) + shape, dtype=jnp.complex64),
        jnp.inf * jnp.ones((freq_band.num)),
    )
    start_step, state, freq_fields, errs = jax.lax.while_loop(
        cond_fn, body_fn, init_foo
    )
    print(f"{errs}, {start_step}, {state.step}, {snapshot_range}")
    return (freq_fields, errs, start_step)

    # TODO: Can we change this into a jax loop?
    for start_step in range(-1, max_steps, steps_per_sim):
        # snapshot_range = Range(
        #     start=start_step
        #     + steps_per_sim
        #     - total_sampling_steps(freq_band, sample_every_n),
        #     interval=sample_every_n,
        #     num=2 * freq_band.num,
        # )
        # wvfrm = waveform

        # wvfrm = waveform[start_step + 1 :]
        # Phase stuff.
        phases = -1 * jnp.pi * jnp.arange(freq_band.num)
        t = (start_step + 1 + jnp.arange(steps_per_sim)) * dt  # TODO: Fix
        phi = freq_band.values[:, None] * t + phases[:, None]
        wvfrm = jnp.sum(jnp.exp(1j * phi), axis=0)

        state = fdtd.State(step=-1, e_field=state.e_field, h_field=state.h_field)

        # Run simulation.
        state, outs = fdtd.simulate(
            dt=dt,
            grid=grid,
            permittivity=permittivity,
            conductivity=conductivity,
            source_field=source,
            source_waveform=wvfrm,  # waveform,
            output_volumes=output_volumes,
            snapshot_range=snapshot_range,
            state=state,
        )

        # Infer time-harmonic fields.
        # TODO: Generalize beyond 1st output.
        t = dt * (
            0.5 + start_step + snapshot_range.interval * jnp.arange(snapshot_range.num)
        )
        freq_fields = sampling.project(outs[0], freq_band, t)

        # Undo phase changes
        freq_fields *= jnp.expand_dims(
            jnp.exp(-1j * phases), axis=range(1, freq_fields.ndim)
        )

        # Compute error.
        errs = wave_equation_error(
            fields=freq_fields,
            freq_band=freq_band,
            permittivity=permittivity,
            conductivity=conductivity,
            source=source,
            grid=grid,
        )
        print(f"{errs}, {start_step}, {state.step}, {snapshot_range}")

        if jnp.max(errs) < err_thresh:
            break

    return (freq_fields, errs, start_step + steps_per_sim)


def wave_equation_error(
    grid: Grid,
    freq_band: Band,
    permittivity: ArrayLike,
    conductivity: ArrayLike,
    source: Subfield,
    fields: ArrayLike,
) -> jax.Array:
    r"""Wave equation error of solution fields.

    Args:
        grid: Same as in :py:func:`solve`.
        freq_band: Same as in :py:func:`solve`.
        permittivity: Same as in :py:func:`solve`.
        conductivity: Same as in :py:func:`solve`.
        source: Same as in :py:func:`solve`.
        fields: ``(ww, 3, xx, yy, zz)`` complex-valued E-fields over the entire
          problem domain corresponding to the angular frequencies denoted by
          ``freq_band``, where ``ww`` corresponds to ``freq_band.num``.

    Returns:
        ``(ww,)`` array of error values (as defined by :py:func:`solve`)
        corresponding to the solutions fields in ``fields``.

    """
    shape = utils.problem_shape(grid, permittivity, conductivity, source)
    w = jnp.expand_dims(freq_band.values, axis=range(-4, 0))

    def operator(u):
        return (
            grids.curl(grids.curl(u, grid, is_forward=True), grid, is_forward=False)
            - (w**2) * (permittivity - 1j * conductivity / w) * u
        )

    def norm(u):
        return jnp.sqrt(jnp.sum(jnp.abs(u) ** 2, axis=range(-4, 0)))

    src = 1j * w * source.field
    err = utils.at(operator(fields), source.offset, source.field.shape[-3:]).add(src)
    return norm(err) / norm(src) / jnp.sqrt(3 * shape[0] * shape[1] * shape[2])


def sampling_strategy(freq_band: Band, permittivity: ArrayLike) -> Tuple(float, int):
    """``(dt, sample_every_n)`` simulation update/extraction parameters."""

    sampling_interval = sampling.sampling_interval(freq_band)

    # Determine the time step to be just under the Courant condition.
    dt = 0.99 * jnp.min(permittivity) / jnp.sqrt(3)

    if freq_band.num == 1:
        # For a single frequency, no need to adjust ``dt``.
        sample_every_n = int(round(sampling_interval / dt))
    else:
        # In the multi-frequency case, adjust ``dt`` to hit
        # ``sampling_interval`` exactly. We do this by reducing ``dt`` to be an
        # integer fraction of  ``sampling_interval``.
        n = int(jnp.floor(sampling_interval / dt))
        dt = sampling_interval / (n + 1)
        sample_every_n = n + 1

    return dt, sample_every_n


def total_sampling_steps(freq_band: Band, sample_every_n: int) -> int:
    """Total number of simulation updates needed to complete sampling."""
    return sample_every_n * (2 * freq_band.num - 1)


def snapshot_range(
    start_step: int, steps_per_sim: int, freq_band: Band, sample_every_n: int
) -> Range:
    """``Range`` of snapshot times for simulation starting on ``start_step``."""
    return Range(
        start=start_step
        + steps_per_sim
        - total_sampling_steps(freq_band, sample_every_n),
        interval=sample_every_n,
        num=2 * freq_band.num,
    )


def simulation_steps(
    freq_band: Band,
    sample_every_n: int,
    shape: Int3,
    min_traversals: float = 1.0,
) -> int:
    """Suggested length of simulations executed in ``solve()``."""
    return max(
        # Number of time steps needed to complete the sampling protocol.
        total_sampling_steps(freq_band, sample_every_n),
        # Allow for ``min_traversals`` bounces along the maximum dimension of
        # the simulation domain, using the crude approximation of traveling a
        # single cell per update step.
        int(min_traversals * max(shape)),
    )


def sim_and_stuff():
    snapshot_range = Range(
        start=-1,
        interval=sample_every_n,
        num=2 * freq_band.num,
    )
    if state is not None:
        state = fdtd.State(step=-1, e_field=state.e_field, h_field=state.h_field)

    wvfrm = waveform[start_step + 1 :]

    # Run simulation.
    state, outs = fdtd.simulate(
        dt=dt,
        grid=grid,
        permittivity=permittivity,
        conductivity=conductivity,
        source_field=source,
        source_waveform=wvfrm,  # waveform,
        output_volumes=output_volumes,
        snapshot_range=snapshot_range,
        state=state,
    )

    # Infer time-harmonic fields.
    # TODO: Generalize beyond 1st output.
    t = dt * (
        0.5 + start_step + snapshot_range.interval * jnp.arange(snapshot_range.num)
    )
    freq_fields = sampling.project(outs[0], freq_band, t)

    # Undo phase changes
    freq_fields *= jnp.expand_dims(
        jnp.exp(-1j * phases), axis=range(1, freq_fields.ndim)
    )

    # Compute error.
    errs = wave_equation_error(
        fields=freq_fields,
        freq_band=freq_band,
        permittivity=permittivity,
        conductivity=conductivity,
        source=source,
        grid=grid,
    )
    print(f"{errs}, {start_step}, {state.step}, {snapshot_range}")
