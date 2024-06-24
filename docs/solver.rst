.. _solver:
.. py:module:: solver
.. currentmodule:: hyperwave.solver

Solver API
==========

Overview
--------

At the heart of hyperwave is the ability to solve Maxwell's equations, which
it accomplishes by running a time-domain simulation under (possibly multiple)
time-harmonic excitations and extracting the resultant fields.

The :py:mod:`solver` module accomplishes this via a general purpose
:py:func:`solve` function which calls the low-level :py:func:`simulate`
function until the termination condition is met.
We leverage a minimalistic implementation of :py:func:`simulate` to determine
the absolute smallest set of features that future performance-optimized
simulation engines will need to implement. 


Primary API
-----------

.. autofunction:: solve
.. autofunction:: wave_equation_error


Secondary API
-------------

.. autoclass:: State
.. autofunction:: simulate

Reference
---------

.. autoclass:: Band
   :members: values
.. autoclass:: Grid
.. autoclass:: Range
.. autoclass:: Subfield
.. autoclass:: Volume
