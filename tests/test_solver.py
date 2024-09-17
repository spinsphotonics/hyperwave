import jax.numpy as jnp
import numpy as np
import pytest

import hyperwave as hw


@pytest.mark.parametrize("num_freqs,shape", [(1, (10, 12, 20))])
def test_field(num_freqs, shape):
    domain = hw.solver.Domain(
        grid=tuple(jnp.ones(s) for s in shape),
        freq_band=hw.solver.FreqBand(0.15, 0.16, num_freqs),
        permittivity=jnp.ones((3,) + shape),
        conductivity=1e-1 * jnp.ones((3,) + shape),
    )
    source = hw.solver.Source(
        values=jnp.zeros((num_freqs, 3, 1, 1, 1)).at[:, 2].set(1.0),
        offset=tuple(s // 2 for s in shape),
    )
    field = hw.solver.field_solve(domain, source)
    np.testing.assert_array_less(hw.solver.field_error(domain, source, field), 1e-3)


@pytest.mark.parametrize(
    "num_freqs,shape,source_offset,source_shape,mode_num,source_axis",
    [(1, (10, 12, 20), (0, 0, 0), (10, 12, 1), 0, 2)],
)
def test_mode(num_freqs, shape, source_offset, source_shape, mode_num, source_axis):
    domain = hw.solver.Domain(
        grid=tuple(jnp.ones(s) for s in shape),
        freq_band=hw.solver.FreqBand(0.15, 0.16, num_freqs),
        permittivity=jnp.ones((3,) + shape),
        conductivity=1e-1 * jnp.ones((3,) + shape),
    )
    source = hw.solver.mode_solve(
        domain, source_offset, source_shape, mode_num, source_axis
    )
    betas = hw.solver.mode_wavevector(domain, source, source_axis)
    np.testing.assert_array_less(
        hw.solver.mode_error(domain, source, source_axis), 1e-3
    )


@pytest.mark.parametrize(
    "num_freqs,shape,axis,pos,is_forward", [(1, (200, 12, 1), 0, 101, True)]
)
def test_cut(num_freqs, shape, axis, pos, is_forward):
    domain = hw.solver.Domain(
        grid=tuple(jnp.ones(s) for s in shape),
        freq_band=hw.solver.FreqBand(0.15, 0.16, num_freqs),
        permittivity=jnp.ones((3,) + shape),
        conductivity=3e-1 * jnp.ones((3,) + shape),
    )
    source = hw.solver.Source(
        values=jnp.zeros((num_freqs, 3, 1, shape[1], shape[2])).at[:, 2].set(1.0),
        offset=(shape[0] // 2, 0, 0),
    )
    field = hw.solver.field_solve(domain, source)
    np.testing.assert_array_less(hw.solver.field_error(domain, source, field), 1e-3)

    cut_field = hw.solver.cut_field(field, axis, pos, is_forward)
    cut_source = hw.solver.cut_solve(domain, field, axis, pos, is_forward)
    np.testing.assert_array_less(
        hw.solver.field_error(domain, cut_source, cut_field), 1e-3
    )
