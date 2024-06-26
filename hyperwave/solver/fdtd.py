# TODO: Change to bfloat16.
"""Implementation of the FDTD method. """

from __future__ import annotations

from typing import NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from . import grids, utils
from .typing import Grid, Range, Subfield, Volume


class State(NamedTuple):
    """Simulation state for the FDTD method.

    Args:
        e_field: ``(3, xx, yy, zz)`` array representing E-field values.
        h_field: ``(3, xx, yy, zz)`` array representing H-field values.

    """

    e_field: ArrayLike
    h_field: ArrayLike

    @staticmethod
    def default(shape, dtype=jnp.float32) -> State:
        return State(
            e_field=jnp.zeros((3,) + shape, dtype=dtype),
            h_field=jnp.zeros((3,) + shape, dtype=dtype),
        )


# Convenience type alias for simulation outputs.
Outputs = Tuple[jax.Array, ...]


def simulate(
    dt: ArrayLike,
    grid: Grid,
    permittivity: ArrayLike,
    conductivity: ArrayLike,
    source_field: Subfield,
    source_waveform: ArrayLike,
    output_volumes: Sequence[Volume],
    snapshot_range: Range,
    state: State | None = None,
) -> Tuple[State, Outputs]:
    r"""Execute the finite-difference time-domain (FDTD) simulation method.

    Uses the convention that for :py:class:`State` at step :math:`i`, the
    :math:`E`-field is at time :math:`(i + 1/2) \Delta t` and the
    :math:`H`-field is at :math:`i \Delta t` (additionally, the current source
    :math:`J`, although spatially located at the :math:`E`-field locations, is
    temporally located at the :math:`H`-field points). Implements the update

    .. math::

        H^{i+1} &= H^i - \Delta t (\nabla \times E^i)

        E^{i+1} &= C_a E^i + C_b (\nabla \times H^{i+1} - J^i)

    where

    .. math::

      z &= \sigma \Delta t / 2 \epsilon'

      C_a &= (1 - z) / (1 + z)

      C_b &= \Delta t / (1 + z) \epsilon'

    and

    * :math:`\epsilon'` is the real-valued permittivity, and
    * :math:`\sigma` is the real-valued conductivity.

    Current source excitation is limited to a fixed complex-valued field pattern
    modulated by a complex-valued waveform such that the injected source is the
    real-part of the product of the field and waveform.

    Only subdomains of the :math:`E`-field are available as outputs, and these
    only at a set of regularly-spaced time points.

    This pure-JAX, feature-minimal implementation of the FDTD method also serves
    to identify the minimum set of features that subsequent simulation engines
    will require in order to serve the :py:func:`solve` API.

    Args:
        dt: Dimensionless value of the amount of time advanced per FDTD update.
        grid: Spacing of the simulation grid.
        permittivity: ``(3, xx, yy, zz)`` array of (relative) permittivity
          values.
        conductivity: ``(3, xx, yy, zz)`` array of conductivity values.
        source_field: ``Subfield`` describing the complex-valued simulation
          input.
        source_waveform: ``(tt,)`` array of complex amplitudes for ``source_field``.
        output_volumes: E-field subvolumes of the simulation space to return.
        snapshot_range: Interval of regularly-spaced time steps at which to
          generate output volumes.
        state: Initial state of the simulation. Defaults to field values of
          ``0`` everywhere at ``step=-1``.

    Returns:
      ``(state, outputs)`` where

      * ``state`` is of type :py:class:`State` and is advanced so as to fulfill
        the most advanced time step requested by ``snapshot_range``, and
      * ``outputs`` is a tuple of ``(nn, 3, xxi, yyi, zzi)`` arrays
        corresponding to the subvolumes identied in ``output_volumes`` and with
        ``nn`` as the number of snapshots as given by ``snapshot_range.num``.

    """
    # Validate inputs.
    shape = utils.problem_shape(grid, permittivity, conductivity, source_field)
    if snapshot_range.start < -1 or snapshot_range.stop != len(source_waveform):
        raise ValueError(
            f"``snapshot_range`` must start after index ``-1`` and must have "
            f"final value corresponding to the last index of "
            f"``source_waveform``, but got got a ``snapshot_range`` of "
            f"{snapshot_range} and ``source_waveform`` of shape "
            f"{source_waveform.shape}."
        )

    # Precomputed update coefficients
    z = conductivity * dt / (2 * permittivity)
    ca = (1 - z) / (1 + z)
    cb = dt / permittivity / (1 + z)

    def source_fn(field: ArrayLike, step: ArrayLike) -> ArrayLike:
        """``field`` with current source at time ``step`` added."""
        return utils.at(field, source_field.offset, source_field.field.shape[-3:]).add(
            -jnp.real(source_field.field * source_waveform[step])
        )

    def step_fn(step, state: State) -> State:
        """``state`` evolved by one FDTD update."""
        e, h = state
        h = h - dt * grids.curl(e, grid, is_forward=True)

        e = ca * e + cb * source_fn(grids.curl(h, grid, is_forward=False), step)
        return State(e, h)

    def output_fn(index: int, outs: Outputs, e_field: ArrayLike) -> Outputs:
        """``outs`` updated at ``index`` with ``e_field``."""
        return tuple(
            out.at[index].set(utils.get(e_field, ov.offset, ov.shape))
            for out, ov in zip(outs, output_volumes)
        )

    def update_and_output(
        state: State, outs: Outputs, output_index: int, lower: int, upper: int
    ) -> Tuple[State, Outputs]:
        """``num_steps`` updates on ``state`` with ``output_index`` snapshot."""
        state = jax.lax.fori_loop(
            lower=lower, upper=upper, body_fun=step_fn, init_val=state
        )
        outs = output_fn(output_index, outs, state.e_field)
        return state, outs

    def update_and_output_for_loop_body(
        output_index: int, state_and_outs: Tuple[State, Outputs]
    ) -> Tuple[State, Outputs]:
        """``update_and_output()`` for ``output_index >= 1``."""
        return update_and_output(
            *state_and_outs,
            output_index,
            lower=snapshot_range[output_index - 1] + 1,
            upper=snapshot_range[output_index] + 1,
        )

    # Initialize ``state`` and ``outs``.
    if state is None:
        state = State.default(shape)
    outs = tuple(jnp.empty((snapshot_range.num, 3) + ov.shape) for ov in output_volumes)

    # Initial update to first output.
    state, outs = update_and_output(
        state, outs, output_index=0, lower=0, upper=snapshot_range[0] + 1
    )

    # Materialize remaining outputs.
    state, outs = jax.lax.fori_loop(
        lower=1,
        upper=snapshot_range.num,
        body_fun=update_and_output_for_loop_body,
        init_val=(state, outs),
    )

    return state, outs
