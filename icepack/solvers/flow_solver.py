# Copyright (C) 2020 by Daniel Shapero <shapero@uw.edu>
#
# This file is part of icepack.
#
# icepack is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found in the file LICENSE in the
# icepack source directory or at <http://www.gnu.org/licenses/>.

r"""Solvers for ice physics models"""

import firedrake
from firedrake import dx, inner, Constant
from icepack.optimization import MinimizationProblem, NewtonSolver
from . import utilities
from ..utilities import default_solver_parameters

# TODO: Remove all dictionary access of 'u' and 'h' once these names are
# fully deprecated from the library


class FlowSolver:
    r"""Solves the diagnostic and prognostic models of ice physics

    The actual solver data is initialized lazily on the first call
    """
    def __init__(self, model, **kwargs):
        self._model = model
        self._fields = {}

        self.dirichlet_ids = kwargs.pop('dirichlet_ids', [])
        self.side_wall_ids = kwargs.pop('side_wall_ids', [])
        self.tolerance = kwargs.pop('tolerance', 1e-12)

        prognostic_parameters = kwargs.get(
            'prognostic_solver_parameters', default_solver_parameters
        )

        if 'prognostic_solver_type' in kwargs.keys():
            solver_type = kwargs['prognostic_solver_type']
            if isinstance(solver_type, str):
                solvers_dict = {
                    'implicit-euler': ImplicitEuler,
                    'lax-wendroff': LaxWendroff
                }
                solver_type = solvers_dict[solver_type]
        else:
            solver_type = LaxWendroff

        self._prognostic_solver = solver_type(
            self.model.continuity, self._fields, prognostic_parameters
        )

    @property
    def model(self):
        r"""The physics model that this object solves"""
        return self._model

    @property
    def fields(self):
        r"""Dictionary of all fields that are part of the simulation"""
        return self._fields

    def _diagnostic_setup(self, **kwargs):
        for name, field in kwargs.items():
            if name in self.fields.keys():
                self.fields[name].assign(field)
            else:
                self.fields[name] = utilities.copy(field)

        # Create homogeneous BCs for the Dirichlet part of the boundary
        u = self.fields.get('velocity', self.fields.get('u'))
        V = u.function_space()
        # NOTE: This will have to change when we do Stokes!
        bcs = None
        if self.dirichlet_ids:
            bcs = firedrake.DirichletBC(V, Constant((0, 0)), self.dirichlet_ids)

        # Find the numeric IDs for the ice front
        boundary_ids = u.ufl_domain().exterior_facets.unique_markers
        ice_front_ids_comp = set(self.dirichlet_ids + self.side_wall_ids)
        ice_front_ids = list(set(boundary_ids) - ice_front_ids_comp)

        # Create the action and scale functionals
        _kwargs = {
            'side_wall_ids': self.side_wall_ids,
            'ice_front_ids': ice_front_ids
        }
        action = self.model.action(**self.fields, **_kwargs)
        scale = self.model.scale(**self.fields, **_kwargs)

        # Set up a minimization problem and solver
        quadrature_degree = self.model.quadrature_degree(**self.fields)
        params = {'quadrature_degree': quadrature_degree}
        problem = MinimizationProblem(action, scale, u, bcs, params)
        self._diagnostic_solver = NewtonSolver(problem, self.tolerance)

    def diagnostic_solve(self, **kwargs):
        r"""Solve the diagnostic model physics for the ice velocity"""
        # Set up the diagnostic solver if it hasn't been already, otherwise
        # copy all the input field values
        if not hasattr(self, '_diagnostic_solver'):
            self._diagnostic_setup(**kwargs)
        else:
            for name, field in kwargs.items():
                if isinstance(field, firedrake.Function):
                    self.fields[name].assign(field)

        # Solve the minimization problem and return the velocity field
        self._diagnostic_solver.solve()
        u = self.fields.get('velocity', self.fields.get('u'))
        return u.copy(deepcopy=True)

    def prognostic_solve(self, dt, **kwargs):
        r"""Solve the prognostic model physics for the new value of the ice
        thickness"""
        return self._prognostic_solver.solve(dt, **kwargs)


class ImplicitEuler:
    def __init__(self, continuity, fields, solver_parameters):
        r"""Prognostic solver implementation using the 1st-order, backward
        Euler timestepping scheme

        This solver is included for backward compatibility only. We do not
        recommend it and the Lax-Wendroff scheme is preferable by far.
        """
        self._continuity = continuity
        self._fields = fields
        self._solver_parameters = solver_parameters

    def setup(self, **kwargs):
        r"""Create the internal data structures that help reuse information
        from past prognostic solves"""
        for name, field in kwargs.items():
            if name in self._fields.keys():
                self._fields[name].assign(field)
            else:
                self._fields[name] = utilities.copy(field)

        dt = firedrake.Constant(1.)
        dh_dt = self._continuity(dt, **self._fields)
        h = self._fields.get('thickness', self._fields.get('h'))
        h_0 = h.copy(deepcopy=True)
        q = firedrake.TestFunction(h.function_space())
        F = (h - h_0) * q * dx - dt * dh_dt

        problem = firedrake.NonlinearVariationalProblem(F, h)
        self._solver = firedrake.NonlinearVariationalSolver(
            problem, solver_parameters=self._solver_parameters
        )

        self._thickness_old = h_0
        self._timestep = dt

    def solve(self, dt, **kwargs):
        r"""Compute the thickness evolution after time `dt`"""
        if not hasattr(self, '_solver'):
            self.setup(**kwargs)
        else:
            for name, field in kwargs.items():
                if isinstance(field, firedrake.Function):
                    self._fields[name].assign(field)

        h = self._fields.get('thickness', self._fields.get('h'))
        self._thickness_old.assign(h)
        self._timestep.assign(dt)
        self._solver.solve()
        return h.copy(deepcopy=True)


class LaxWendroff:
    def __init__(self, continuity, fields, solver_parameters):
        r"""Prognostic solver implementation using the 2st-order implicit
        Lax-Wendroff timestepping scheme

        This method introduces additional diffusion along flowlines compared
        to the implicit Euler scheme. This tends to reduce the magnitude of
        possible spurious oscillations.
        """
        self._continuity = continuity
        self._fields = fields
        self._solver_parameters = solver_parameters

    def setup(self, **kwargs):
        r"""Create the internal data structures that help reuse information
        from past prognostic solves"""
        for name, field in kwargs.items():
            if name in self._fields.keys():
                self._fields[name].assign(field)
            else:
                self._fields[name] = utilities.copy(field)

        dt = firedrake.Constant(1.)
        h = self._fields.get('thickness', self._fields.get('h'))
        u = self._fields.get('velocity', self._fields.get('u'))
        h_0 = h.copy(deepcopy=True)

        Q = h.function_space()
        model = self._continuity
        n = model.facet_normal(Q.mesh())
        outflow = firedrake.max_value(0, inner(u, n))
        inflow = firedrake.min_value(0, inner(u, n))

        # Additional streamlining terms that give 2nd-order accuracy
        q = firedrake.TestFunction(Q)
        div, grad, ds = model.div, model.grad, model.ds
        flux_cells = -div(h * u) * inner(u, grad(q)) * dx
        flux_out = div(h * u) * q * outflow * ds
        flux_in = div(h_0 * u) * q * inflow * ds
        d2h_dt2 = flux_cells + flux_out + flux_in

        dh_dt = model(dt, **self._fields)
        F = (h - h_0) * q * dx - dt * (dh_dt + 0.5 * dt * d2h_dt2)

        problem = firedrake.NonlinearVariationalProblem(F, h)
        self._solver = firedrake.NonlinearVariationalSolver(
            problem, solver_parameters=self._solver_parameters
        )

        self._thickness_old = h_0
        self._timestep = dt

    def solve(self, dt, **kwargs):
        r"""Compute the thickness evolution after time `dt`"""
        if not hasattr(self, '_solver'):
            self.setup(**kwargs)
        else:
            for name, field in kwargs.items():
                if isinstance(field, firedrake.Function):
                    self._fields[name].assign(field)

        h = self._fields.get('thickness', self._fields.get('h'))
        self._thickness_old.assign(h)
        self._timestep.assign(dt)
        self._solver.solve()
        return h.copy(deepcopy=True)
