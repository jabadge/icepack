# Copyright (C) 2021 by Andrew Hoffman <hoffmaao@uw.edu>
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
from ..utilities import default_solver_parameters



class FirnSolver:    
    def __init__(self,model, **kwargs):
        r"""Solves the firn models of snow densification physics

        This class is responsible for efficiently solving the physics
        problem you have chosen. (This is contrast to classes like
        Firn, which is where you choose what that physics problem
        is.) If you want to make your simulation run faster, you can select
        different solvers and options.

        Parameters
        ----------
        model
            The firn model object -- currently just Harron-Langway model.
        dirichlet_ids : list of int, optional
            Numerical IDs of the boundary segments where the ice velocity
            should be fixed
        side_wall_ids : list of int, optional
            Numerical IDs of the boundary segments where the ice velocity
            should have no normal flow
        densification_solver_type : {'icepack', 'petsc'}, optional
            Use hand-written optimization solver ('icepack') or PETSc SNES
            ('petsc'), defaults to 'icepack'
        densification_solver_parameters : dict, optional
            Options for the diagnostic solver; defaults to a Newton line
            search method with direct factorization of the Hessian using
            MUMPS

        Examples
        --------

        Create a flow solver with inflow on boundary segments 1 and 2
        using the default solver configuration.

        >>> model = icepack.models.Firn()
        >>> solver = icepack.solvers.FlowSolver(model, dirichlet_ids=[1, 2])

        Use an iterative linear solver to hopefully accelerate the code.

        >>> opts = {
        ...     'dirichlet_ids': [1, 2],
        ...     'diagnostic_solver_type': 'petsc',
        ...     'diagnostic_solver_parameters': {
        ...         'ksp_type': 'cg',
        ...         'pc_type': 'ilu',
        ...         'pc_factor_fill': 2
        ...     },
        ...     'prognostic_solver_parameters': {
        ...         'ksp_type': 'gmres',
        ...         'pc_type': 'sor'
        ...     }
        ... }
        >>> solver = icepack.solvers.FlowSolver(model, **opts)
        """
        self._model = model
        self._fields = {}

        self._solver_parameters = kwargs.get(
            'densification_solver_parameters', default_solver_parameters
        )
    @property
    def model(self):
        r"""The frin model that this object solves"""
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
        
        dt = Constant(1.)
        
        ρflux = self.model.density_gradient_flux(**self.fields)
        Δρ = self.model.densification(**self.fields)
        wflux = self.model.velocity_gradient_flux(**self.fields)
        ρvel = self.model.densification_velocity(**self.fields)
        
        dρ_dt = ρflux - Δρ
        
        ρ = self.fields.get('density', self.fields.get('ρ'))
        w = self.fields.get('firn_velocity', self.fields.get('w_f'))
        h_f = self.fields.get('firn_thickness',self.fields.get('h_f'))
        ρ_s = self.fields.get('surface_density', self.fields.get('ρ_s'))
        w_s = self.fields.get('accumulation', self.fields.get('a'))
        ρ_0 = ρ.copy(deepcopy=True)
        w_0 = w.copy(deepcopy=True)
        
        ϕ = firedrake.TestFunction(ρ.function_space())
        
        F = (ρ - ρ_0) * ϕ * h_f * dx - dt * dρ_dt
        G = wflux + ρvel

        self._ρbcs = firedrake.DirichletBC(ρ.function_space().sub(0), ρ_s,'top')
        self._wbcs = firedrake.DirichletBC(w.function_space().sub(0), w_s,'top')

        degree = ρ.ufl_element().degree()
        fc_params = {'quadrature_degree': (3 * degree[0], 2 * degree[1])}
        
        densityproblem = firedrake.NonlinearVariationalProblem(
            F, ρ, form_compiler_parameters=fc_params,bcs=self._ρbcs
        )
        velocityproblem = firedrake.NonlinearVariationalProblem(
            G, w, form_compiler_parameters=fc_params,bcs=self._wbcs
        )
        

        self._density_solver = firedrake.NonlinearVariationalSolver(
            densityproblem, solver_parameters=self._solver_parameters
        )
        self._velocity_solver = firedrake.NonlinearVariationalSolver(
            velocityproblem, solver_parameters=self._solver_parameters
        )

        self._density_old = ρ_0
        self._velocity_old = w_0
        self._timestep = dt
        
    def solve_density(self, dt, **kwargs):
        if not hasattr(self, '_density_solver'):
            self._setup(**kwargs)
        else:
            for name, field in kwargs.items():
                self.fields[name].assign(field)

        ρ = self.fields.get('density', self.fields.get('ρ'))
        self._density_old.assign(ρ)
        self._timestep.assign(dt)
        self._density_solver.solve()
        return ρ.copy(deepcopy=True)

    def solve_velocity(self,dt, **kwargs):
        if not hasattr(self, '_velocity_solver'):
            self._setup(**kwargs)
        else:
            for name, field in kwargs.items():
                self.fields[name].assign(field)

        w = self.fields.get('firn_velocity',self.fields.get('w'))
        self._velocity_old.assign(w)
        self._timestep.assign(dt)
        self._velocity_solver.solve()
        return w.copy(deepcopy=True)