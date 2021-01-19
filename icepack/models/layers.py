# Copyright (C) 2018-2019 by Daniel Shapero <shapero@uw.edu> and Andrew Hoffman <hoffmaao@uw.edu>
#
# This file is part of Andrew Hoffman's development branch of icepack.
#
# icepack is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found in the file LICENSE in the
# icepack source directory or at <http://www.gnu.org/licenses/>.

"""
Solver for internal layers representing isochronal structures
"""

import firedrake
import icepack.models.viscosity
import math
import numpy as np
from firedrake import (
    grad,
    dx,
    ds,
    dS,
    sqrt,
    Identity,
    inner,
    sym,
    tr as trace,
    det,
    dot,
    div,
    lt,
)
from icepack.constants import year, glen_flow_law as n

def orthogonal_velocity(u,h,mesh,W):
    """
    Solve for velocity orthogonal to bed
    ----------
    u : firedrake.Function
        Ice velocity
    h : firedrake.Function
        Ice thickness
    mesh: firedrake.mesh
        model mesh object
    W : firedrake.FunctionSpace
        vertical velocity function space
    """
    x, y, ζ = firedrake.SpatialCoordinate(mesh)
    ω_expr = -(u[0].dx(0) + u[1].dx(1)) / h * ζ
    ω = firedrake.project(ω_expr, W)
    w = firedrake.interpolate(h * ω, W)
    return w

class slopes(object):
    def solve(self, u, h, s, mesh):
        """
        Solve for steady state layer slopes in radians
        ----------
        u : firedrake.Function
            Ice velocity
        h : firedrake.Function
            Ice thickness
        s : firedrake.Function
            Ice surface elevation
        mesh : firedrake.mesh
            model mesh object
        #W : firedrake.FunctionSpace
        #    vertical velocity function space
        """
        V = u.function_space()
        W = h.function_space()
        w = orthogonal_velocity(u,h,mesh,W)
        slopes = firedrake.interpolate(firedrake.as_vector((firedrake.atan(w/u[0]),firedrake.atan(w/u[1]))),V)
        return slopes
