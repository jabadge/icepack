# Copyright (C) 2021 by Jessica Badgeley <badgeley@uw.edu> and
# Andrew Hoffman <hoffmaao@uw.edu>
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

import firedrake
from firedrake import (
    interpolate,
    conditional,
    Constant,
    dx,
    assemble,
)
import icepack
import numpy as np
import matplotlib.pyplot as plt

# Test our numerical solver against a simple, analytical solution.
num_years = 10.0
Lx, Ly = 1.e3, 1.e3
ux = Constant(40.)
uy = Constant(0.)
uz = Constant(-40.)
dts = [1/5.0, 1/6.0, 1/7.0, 1/8.0, 1/9.0]
dx_constant = 50.
dxs = [100., 75., 50., 40., 30.]
dt_constant = 3./25.
age0 = 0.

def velocity_3D(**kwargs):
    u = kwargs['velocity']
    h = kwargs['thickness']
    uz = kwargs['uz']
    Q = h.function_space()
    V = u.function_space()
    mesh = Q.mesh()
    xdegree_u, zdegree_u = u.ufl_element().degree()
    W = firedrake.FunctionSpace(mesh, family='CG', degree=2, vfamily='GL', vdegree=1+V.ufl_element().degree()[1])
    w = firedrake.interpolate(uz/h,W)
    V3D = firedrake.VectorFunctionSpace(mesh, dim=3, family='CG', degree=xdegree_u, vfamily='GL',vdegree=zdegree_u+1)
    u3D = firedrake.Function(V3D).interpolate(firedrake.as_vector((u[0],u[1],w)))
    return u3D

def create_mask(x, y, ζ, h):
    x_min = ux.dat.data[0] * num_years
    y_min = uy.dat.data[0] * num_years
    ζ_max = (h.dat.data[0] + uz.dat.data[0] * num_years) / h.dat.data[0]
    mask_x = conditional(x < x_min, 0, 1)
    mask_y = conditional(y < y_min, 0, 1)
    mask_ζ = conditional(ζ > ζ_max, 0, 1)
    return mask_x * mask_y * mask_ζ

def create_mesh_function_spaces(nx, ny):
    mesh2d = firedrake.RectangleMesh(nx, ny, Lx, Ly)
    mesh = firedrake.ExtrudedMesh(mesh2d, layers=1)
    Q = firedrake.FunctionSpace(mesh, family='CG', degree=2, vfamily='DG', vdegree=0)
    V = firedrake.VectorFunctionSpace(mesh, dim=2, family='CG', degree=2, vfamily='GL', vdegree=6)
    Q3D = firedrake.FunctionSpace(mesh, family='DG', degree=2, vfamily='GL', vdegree=1+V.ufl_element().degree()[1])
    return mesh, Q, V, Q3D

def create_variables(Q, V, Q3D):
    b = interpolate(firedrake.Constant(0.0), Q)
    s = interpolate(firedrake.Constant(Lx), Q)
    h = interpolate(s - b, Q)
    u = interpolate(firedrake.as_vector((ux, uy)), V)
    w = interpolate(uz/h,Q3D)
    return b, s, h, u, w

def norm(v):
    return icepack.norm(v, norm_type="L2")

def test_diagnostic_solver_convergence_time():
    nx, ny = int(Lx/dx_constant), int(Ly/dx_constant)
    mesh, Q, V, Q3D = create_mesh_function_spaces(nx, ny)
    b, s, h, u, w = create_variables(Q, V, Q3D)
    x, y, ζ = firedrake.SpatialCoordinate(mesh)
    mask = create_mask(x, y, ζ, h)
    u3D = velocity_3D(velocity=u, thickness=h, uz=uz)
    age_model = icepack.models.AgeTransport(velocity_3D=velocity_3D)
    age_solver = icepack.solvers.AgeSolver(age_model)
    
    errors = []

    for dt in dts:
        
        age = interpolate(Constant(age0),Q3D) 
        num_timesteps = int(num_years / dt)

        for step in range(num_timesteps):
            
            t = step*dt
            age = age_solver.solve(dt,velocity=u,thickness=h,age=age,uz=uz)            
        age_num = interpolate(age * mask, Q3D)
        age_exact = interpolate((age0 + t) * mask, Q3D)

        #errors = [assemble((age_exact - age_num) * dx)/assemble(age_exact * dx)
        #errors += [assemble((age_num - age_exact) * dx)]
        errors += [norm(age_exact - age_num) / norm(age_exact)]

    print(dts)
    print(errors)    

    plt.figure()
    plt.plot(dts, errors, 'ko')
    plt.show()

    log_dt = np.log2(np.array(dts))
    log_error = np.log2(np.array(errors))
    print(log_dt)
    print(log_error)
    slope, intercept = np.polyfit(log_dt, log_error, 1)

    plt.figure()
    plt.plot(log_dt, log_error, 'ko')
    plt.show()

    print(f"error ~= {slope:g} * dt + {intercept:g}")
    assert slope > 0

def test_diagnostic_solver_convergence_space():
    
    num_timesteps = int(num_years / dt_constant)
    errors = []

    for dx_val in dxs:
    
        nx, ny = int(Lx/dx_val), int(Ly/dx_val)
        mesh, Q, V, Q3D = create_mesh_function_spaces(nx, ny)
        b, s, h, u, w = create_variables(Q, V, Q3D)
        x, y, ζ = firedrake.SpatialCoordinate(mesh)
        mask = create_mask(x, y, ζ, h)
        u3D = velocity_3D(velocity=u, thickness=h, uz=uz)
        age_model = icepack.models.AgeTransport(velocity_3D=velocity_3D)
        age_solver = icepack.solvers.AgeSolver(age_model)   

        age = interpolate(Constant(age0),Q3D)

        for step in range(num_timesteps):

            t = dt_constant * step
            age = age_solver.solve(dt_constant,velocity=u,thickness=h,age=age,uz=uz)

        age_num = interpolate(age * mask, Q3D)
        age_exact = interpolate((age0 + t) * mask, Q3D)

        #errors = [assemble((age_exact - age_num) * dx)/assemble(age_exact * dx)
        errors += [assemble((age_num - age_exact) * dx)]
        #errors += [norm(age_exact - age_num) / norm(age_exact)]

    print(dxs)
    print(errors)    

    plt.figure()
    plt.plot(dxs, errors, 'ko')
    plt.show()

    log_dx = np.log2(np.array(dxs))
    log_error = np.log2(np.array(errors))
    print(log_dx)
    print(log_error)
    slope, intercept = np.polyfit(log_dx, log_error, 1)

    plt.figure()
    plt.plot(log_dx, log_error, 'ko')
    plt.show()

    print(f"error ~= {slope:g} * dx + {intercept:g}")
    assert slope > 0
