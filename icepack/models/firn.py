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

import functools
import sympy
import firedrake
from firedrake import (
    grad, exp, inner, outer, sym, Identity, tr as trace, sqrt, dx, ds_b, ds_v
)

from icepack.models.mass_transport import Continuity
from icepack.optimization import MinimizationProblem, NewtonSolver
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    gravity as g,
    ideal_gas as R,
    act_grain_growth as Eg,
    act_water_ice as Ec,
    clausius_clapeyron as beta,
    creep_coefficient_high as kcHh,
    creep_coefficient_low as kcLw,
    grain_growth_coefficient as kg,
    heat_capacity as cpi,
    year as year
)
from icepack.utilities import (
	eigenvalues,
    facet_normal_2,
    grad_2,
    add_kwarg_wrapper,
    get_kwargs_alt
)

### TODO: Make this dependent on gradient/search and find the z-dimension of interval mesh.


def HarronLangway(ρ_crit=550/ year**2 * 1.0e-6, k1=11.0, k2=575.0, Q1=10.16, Q2=21.4, aHL=1.0, bHL=0.5,**kwargs):
    r""" Harron-Langway densification model
    Need to understand the units of k1 and k2 and make sure these are consistent with the model units.
    """
    keys = ('density', 'accumulation', 'temperature')
    keys_alt = ('ρ', 'a', 'T')
    ρ, a, T = get_kwargs_alt(kwargs, keys, keys_alt)
    c = firedrake.conditional(ρ<=ρ_crit,k1*exp(-Q1/(R*T))*a**aHL,k2*exp(-Q2/(R*T))*a**bHL)
    dρdt = c*(ρ_I-ρ)
    
    return dρdt


class FirnModel:
    r"""Class for modeling depth dependent density profile
        
    This class solves for the firn density and vertical velocity (additional 
    energy density solutions are also in the developmental scope of the model).
    This will allow a thermodynamically consistent derivation of the firn density.
    Alternative models for the densification rate can be supplied via dρdt.
    """

    def __init__(self, dρdt=HarronLangway):
    	self.dρdt=add_kwarg_wrapper(dρdt)
    
    def density_gradient_flux(self, **kwargs):
        r"""Return the density gradient contribution to density residual

        Parameters
        ----------
        h_f : firedrake.Function
            firn thickness
        w_f : firedrake.Function
            firn velocity
        ρ   : firedrake.Function
            firn density
        a   : firdrake.Function
            surface accumulation
        """
        keys = ('firn_thickness','firn_velocity', 'density', 'accumulation')
        keys_alt = ('h_f', 'w_f', 'ρ', 'a')
        
        h_f, w, ρ, a = get_kwargs_alt(kwargs, keys, keys_alt)
                
        Q = ρ.function_space()
        ϕ = firedrake.TestFunction(Q)
        
        return w*ρ.dx(2)*ϕ*h_f*dx

    def densification(self, **kwargs):
        r"""Return the densification part of the density residual

        This form solves for the \frac{dρ}{dt} component of the
        density residual, which is often the term varied across different
        firn models. 

        Parameters
        ----------
        h_f : firedrake.Function
            firn thickness
        ρ   : firedrake.Function
            frin density
        T_f : firdrake.Function
            firn temperature
        a   : firedrake.Function
            surface accumulation
        """

        keys = ('firn_thickness', 'density', 'temperature', 'accumulation')
        keys_alt = ('h_f','ρ','T_f','a')
        h_f, ρ, T, a = get_kwargs_alt(kwargs, keys, keys_alt)
        dρdt = self.dρdt(**kwargs)
            
        Q = ρ.function_space()
        ϕ = firedrake.TestFunction(Q)
            
        return dρdt*ϕ*dx

    def velocity_gradient_flux(self, **kwargs):
        r"""Return the velocity gradient part of the 
        veloctiy residual

        Parameters
        ----------
        h_f : firedrake.Function
            firn thickness
        w_f : firedrake.Function
            firn densification velocity
        ρ   : firedrake.Function
            frin density
        """
        keys = ('firn_thickness','firn_velocity','density')
        keys_alt = ('h_f','w_f','ρ')
        h_f, w, ρ = get_kwargs_alt(kwargs, keys, keys_alt)
            
        Q = w.function_space()
        η = firedrake.TestFunction(Q)

        return ρ*w.dx(2)*η*h_f*dx

    def densification_velocity(self, **kwargs):
        r"""Return the densification velocity part of 
        the density residual

        Parameters
        ----------
        h_f : firedrake.Function
            firn thickness
        w_f : firedrake.Function
            firn densification velocity
        ρ   : firedrake.Function
            frin density
        T_f : firedrake.Function
            firn temperature
        a   : firedrake.Function
            surface accumulation
        """

        keys = ('firn_thickness','firn_velocity','density','firn_temperature','accumulation')
        keys_alt = ('h_f','w_f','ρ','T_f','a')
        h_f, w, ρ ,T,a = get_kwargs_alt(kwargs, keys, keys_alt)
        dρdt = self.dρdt(**kwargs)
            
        Q = w.function_space()
        η = firedrake.TestFunction(Q)
            
        return dρdt*η*h_f*dx