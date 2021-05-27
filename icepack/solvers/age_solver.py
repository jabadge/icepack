# Copyright (C) 2020-2021 by Andrew Hoffman <hoffmaao@uw.edu>
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

r"""Description of the age solver used to calculate the layer slopes
"""

import firedrake
from firedrake import (
    dx,
    LinearVariationalProblem,
    LinearVariationalSolver,
    max_value,
    min_value
)

class AgeSolver:
    def __init__(self, model):
        self._model = model
        self._fields = {}
    @property
    def model(self):
        r"""The heat transport model that this object solves"""
        return self._model
    @property
    def fields(self):
        r"""The dictionary of all fields that are part of the simulation"""
        return self._fields
    def _setup(self, **kwargs):
        for name, field in kwargs.items():
            if name in self._fields.keys():
                self._fields[name].assign(field)
            else:
                if isinstance(field, firedrake.Constant):
                    self._fields[name] = firedrake.Constant(field)
                elif isinstance(field, firedrake.Function):
                    self._fields[name] = field.copy(deepcopy=True)
                else:
                    raise TypeError('Input fields must be Constant or Function!')

        # Create symbolic representations of the flux and sources of damage
        dt = firedrake.Constant(1.0)
        flux = self.model.flux(**self.fields)

        # Create the finite element mass matrix
        q = self.fields.get('age', self.fields.get('age'))
        Q = q.function_space()
        φ, ψ = firedrake.TrialFunction(Q), firedrake.TestFunction(Q)
        M = φ * ψ * dx

        L1 = -dt * flux
        q1 = firedrake.Function(Q)
        q2 = firedrake.Function(Q)
        L2 = firedrake.replace(L1, {q: q1})
        L3 = firedrake.replace(L1, {q: q2})

        dq = firedrake.Function(Q)
        
        parameters = {
            'solver_parameters': {
                'ksp_type': 'preonly',
                'pc_type': 'bjacobi',
                'sub_pc_type': 'ilu'
            }
        }

        problem1 = LinearVariationalProblem(M, L1, dq)
        problem2 = LinearVariationalProblem(M, L2, dq)
        problem3 = LinearVariationalProblem(M, L3, dq)
        solver1 = LinearVariationalSolver(problem1, **parameters)
        solver2 = LinearVariationalSolver(problem2, **parameters)
        solver3 = LinearVariationalSolver(problem3, **parameters)

        self._solvers = [solver1, solver2, solver3]
        self._stages = [q1, q2]
        self._age_change = dq
        self._timestep = dt
        
    def solve(self, dt, **kwargs):
        if not hasattr(self, '_solvers'):
            self._setup(**kwargs)
        else:
            for name, field in kwargs.items():
                self.fields[name].assign(field)

        δt = self._timestep
        δt.assign(dt)
        q = self.fields.get('age', self.fields.get('age'))

        solver1, solver2, solver3 = self._solvers
        q1, q2 = self._stages
        dq = self._age_change

        solver1.solve()
        q1.assign(q + dq)
        solver2.solve()
        q2.assign(3/4 * q + 1/4 * (q1 + dq))
        solver3.solve()
        q.assign(1/3 * q + 2/3 * (q2 + dq))
        
        max_age = firedrake.Constant(self.model.max_age)
        q.project(min_value(max_value(q + firedrake.Constant(δt), firedrake.Constant(0.0)), max_age))
        return q.copy(deepcopy=True)




