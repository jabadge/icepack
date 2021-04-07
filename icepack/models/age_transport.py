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

r"""Description of the age transport model.
"""

import firedrake
from firedrake import (
    inner, grad, div, dx, ds, ds_t, ds_v, dS_v, dS_h, dS, sqrt, det, min_value, max_value, conditional
)
from icepack.constants import year
from icepack.utilities import eigenvalues, vertical_velocity, get_kwargs_alt


class AgeTransport:
    def __init__(
        self,
        max_age=2000000.0
    ):
        self.max_age = max_age


        
    def flux(self, **kwargs):
        keys = ('thickness', 'velocity', 'age')
        keys_alt = ('h','u','age')
        h, u, q = get_kwargs_alt(kwargs, keys, keys_alt)

        Q = h.function_space()
        V = u.function_space()
        mesh = Q.mesh()
        U = q.function_space()
        φ = firedrake.TestFunction(U)

        w = firedrake.interpolate(vertical_velocity(u,h),U)

        xdegree_u, zdegree_u = u.ufl_element().degree()
        degree_h = h.ufl_element().degree()[0]
        degree = (xdegree_u + degree_h, 2 * zdegree_u + 1)
        metadata = {'quadrature_degree': degree}
        ice_front_ids = tuple(kwargs.pop('ice_front_ids', ()))
        ds_terminus = ds_v(domain=mesh, subdomain_id=ice_front_ids, metadata=metadata)
        V3D = firedrake.VectorFunctionSpace(mesh,dim=3, family='CG',degree=xdegree_u,vfamily='GL',vdegree=zdegree_u+1)
        u3D = firedrake.Function(V3D).interpolate(firedrake.as_vector((u[0],u[1],w)))

        n = firedrake.FacetNormal(mesh)
        u_n = max_value(0, inner(u3D, n))
        f = q * u_n
        q_inflow=firedrake.Constant(0.0)
        flux_faces_v = (f('+') - f('-')) * (φ('+') - φ('-')) * dS_v
        flux_faces_h = (f('+') - f('-')) * (φ('+') - φ('-')) * dS_h
        flux_cells = -q * div(u3D * φ) * dx
        flux_terminus = q * max_value(0, inner(u3D, n)) * φ * ds_terminus
        flux_surface = q_inflow * min_value(0, inner(u3D, n)) * φ * ds_t
        return flux_cells + flux_terminus + flux_surface + flux_faces_v + flux_faces_h
