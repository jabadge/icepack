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

r"""Description of the continuum damage mechanics model
This module contains a solver for the conservative advection equation that
describes the evolution of ice damage (Albrecht and Levermann 2014).
"""

import firedrake
from firedrake import (
    inner, grad, div, dx, ds, ds_t, dS, sqrt, det, min_value, max_value, conditional
)
from icepack.constants import year
from icepack.utilities import eigenvalues, get_kwargs_alt



class AgeTransport:
    def vertical_velocity(u,h):
        """
        Solve for velocity orthogonal to bed
        ----------
        u : firedrake.Function
            Ice velocity
        h : firedrake.Function
            Ice thickness
        """
        Q=h.functionspace()
        V=u.functionspace()
        mesh=Q.mesh()

        xdegree_u, zdegree_u = u.ufl_element().degree()
        W = firedrake.FunctionSpace(mesh,family='CG',degree=xdegree_u,vfamily='GL',vdegree=zdegree_u)

        x, y, ζ = firedrake.SpatialCoordinate(mesh)
        w_expr = -(u[0].dx(0) + u[1].dx(1)) * ζ

        return firedrake.interpolate(w_expr,W)


    def flux(self, **kwargs):
        keys = ('thickness', 'velocity')
        keys_alt = ('h', 'u')
        h, u = get_kwargs_alt(kwargs, keys, keys_alt)
        
        w = vertical_velocity(u,h)

        q_inflow = firedrake.Constant(0.0)

        Q = q.function_space()
        φ = firedrake.TestFunction(Q)

        xdegree_u, zdegree_u = u.ufl_element().degree()
        degree_h = h.ufl_element().degree()[0]
        degree = (xdegree_u + degree_h, 2 * zdegree_u + 1)
        metadata = {'quadrature_degree': degree}
        ice_front_ids = tuple(kwargs.pop('ice_front_ids', ()))
        ds_terminus = ds_v(domain=mesh, subdomain_id=ice_front_ids, metadata=metadata)

        V3 = firedrake.VectorFuncgionSpace(mesh,dim=3, family='CG',degree=degree_u,vfamily='GL',vdegree=zdegree_u)
        u3D = firedrake.Function(V3D).interpolate(firedrake.as_vector((u[0],u[1],w)))

        mesh = Q.mesh()
        n = firedrake.FacetNormal(mesh)
        u_n = max_value(0, inner(u3D, n))
        f = q * u_n

        flux_faces = (f('+') - f('-')) * (φ('+') - φ('-')) * dS
        flux_cells = -q * div(u3D * φ) * dx
        flux_out = q * max_value(0, inner(u3D, n)) * φ * ds_terminus
        flux_in = q_inflow * min_value(0, inner(u3D, n)) * φ * ds_t
        source =  φ*ds

        return flux_faces + flux_cells + flux_out + flux_in + source
