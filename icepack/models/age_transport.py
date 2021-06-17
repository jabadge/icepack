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

from operator import itemgetter
import firedrake
from firedrake import (
    inner, grad, div, dx, ds, ds_tb, ds_v, dS_v, dS_h, dS, sqrt, det, min_value, max_value, conditional
)
from icepack.constants import year
from icepack.utilities import eigenvalues, vertical_velocity, add_kwarg_wrapper

def velocity_3D(**kwargs):
    u, h = itemgetter("velocity", "thickness")(kwargs)

    Q = h.function_space()
    V = u.function_space()
    mesh = Q.mesh()
    xdegree_u, zdegree_u = u.ufl_element().degree()
    W = firedrake.FunctionSpace(mesh, family='CG', degree=2, vfamily='GL', vdegree=1+V.ufl_element().degree()[1])
    w = firedrake.interpolate(vertical_velocity(u,h),W)
    V3D = firedrake.VectorFunctionSpace(mesh, dim=3, family='CG', degree=xdegree_u, vfamily='GL',vdegree=zdegree_u+1)
    u3D = firedrake.Function(V3D).interpolate(firedrake.as_vector((u[0],u[1],w)))
    return u3D 


class AgeTransport:
    def __init__(
        self,
        max_age=2000000.0,
        velocity_3D=velocity_3D
    ):
        self.max_age = max_age
        self.velocity_3D = add_kwarg_wrapper(velocity_3D)


        
    def flux(self, **kwargs):
        u, h, q = itemgetter("velocity", "thickness", "age")(kwargs)

        u3D = self.velocity_3D(**kwargs)

        Q = h.function_space()
        U = q.function_space()
        mesh = Q.mesh()
        
        φ = firedrake.TestFunction(U)

        xdegree_u, zdegree_u = u.ufl_element().degree()
        degree_h = h.ufl_element().degree()[0]
        degree = (xdegree_u + degree_h, 2 * zdegree_u + 1)
        metadata = {'quadrature_degree': degree}
        ice_front_ids = tuple(kwargs.pop('ice_front_ids', ()))

        u3D = self.velocity_3D(**kwargs)

        n = firedrake.FacetNormal(mesh)
        u_n = max_value(0, inner(u3D, n))
        f = q * u_n
        q_inflow=firedrake.Constant(0.0)
        flux_faces_v = (f('+') - f('-')) * (φ('+') - φ('-')) * dS_v
        flux_cells = -q * div(u3D * φ) * dx
        flux_v = q * max_value(0, inner(u3D, n)) * φ * ds_v
        flux_tb = q * max_value(0, inner(u3D, n)) * φ * ds_tb
        return flux_cells + flux_faces_v + flux_tb + flux_v
