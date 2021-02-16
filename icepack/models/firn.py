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
    max_value, grad, exp, log, inner, outer, sym, Identity, tr as trace, sqrt, dx, ds_b, ds_v
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
    melting_temperature as mt,
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


def density2stress(ρ):
    # We need a density to stress function of some kind.
    return None
def fac(ρ,h_f):
    # firn air content
    return None

def melt_water_content(ρ,h_f):
    # melt water content for percolation
    return None

def HerronLangway(ρ, a, T, ρ_crit=550.0, k1=11.0, k2=575.0, Q1=10.16, Q2=21.4, aHL=1.0, bHL=0.5,**kwargs):
    r""" Herron-Langway densification model
    Need to understand the units of k1 and k2 and make sure
    these are consistent with the model units.
    """
    c = firedrake.conditional(ρ<ρ_crit,k1*exp(-Q1/(R*T))*(a*ρ_I/ρ_W)**aHL,k2*exp(-Q2/(R*T))*(a*ρ_I/ρ_W)**bHL)
    dρdt = c*((ρ_I * year**2 / 1.0e-6)-ρ)
    return dρdt


def HL_Sigfus(ρ, a, T, ρ_crit=550.0, k1=11.0, k2=575.0, Q1=10.16, Q2=21.4, aHL=1.0, bHL=0.5, **kwargs):
    r""" Sigfus implementation of the Herron-Langway 
    """

    σ=density2stress(ρ)
    dρ=(ρ_I * year**2 / 1.0e-6 - ρ) / 1000.0
    dσ=σ-σ_crit
    ksig=k2 * exp(-Q2/(R*T))**2
    c=firedrake.conditional(ρ<ρ_crit,k1*exp(-Q1/(R*T))*(a*ρ_I/ρ_W)**aHL,ksig*dσ/(g*log((ρ_I/ρ_W -ρ_crit/ρ_W))))
    dρdt=firedrake.condition(ρ< _I* year**2 / 1.0e-6,c*((ρ_I * year**2 / 1.0e-6)-ρ),0.0)

    return None


def Arthern_2010T(ρ, a, T, r, ρ_crit=550.0, kc1=9.2e-9, kc2=3.7e-9,**kwargs):
    r""" Arthern's tarnsient model described in the
    appendix of Arthern et al. 2010. Uses stress
    rather than accumulation rate.
    """
    σ=density2stress(ρ)

    c=firedrake.conditional(ρ<ρ_crit,kc1*exp(-Ec/(R*T))*σ/r,kc2*exp(-Ec/(R*T))*σ/r)
    dρdt = c*((ρ_I * year**2 / 1.0e-6)-ρ)
    return None

def Helsen_2008(ρ, a, T, ρ_crit=550.0, k1=11.0, k2=575.0, Q1=10.16, Q2=21.4, aHL=1.0, bHL=0.5, **kwargs):
    r""" Helsen 2008 implementation of equation
    from Arthern et al. 2010.
    """

    Tavg = utilities.lift3d(utilities.depth_average(T))
    c=a*ρ_I/ρ_W * ( 76.138 - 0.28965 * Tavg * 8.36 * (mt - T)**-2.061) 
    dρdt=c*((ρ_I * year**2 / 1.0e-6) - ρ)

    return dρdt

def Li_2004(ρ, a, T, **kwargs):
    r""" Accumulation units are m W.E. per year
    Equation is published in Artherm 2010
    requires vapor flux to be implemented properly.
    """

    Tavg = utilities.lift3d(utilities.depth_average(T))
    c = a*ρ_I/ρ_W*(139.21-0.542*Tavg)*8.36*(mt-T)**-2.061
    dρdt=c*((ρ_I * year**2 / 1.0e-6) - ρ)
    
    return dρdt

def Li_2011(ρ, a, T, **kwargs):
    r""" 
    """

    Tavg = utilities.lift3d(utilities.depth_average(T))
    β1 = -9.788 + 8.996*a*ρ_I/ρ_W - 0.6165*(Tavg-mt)
    β2 = β1/(-2.0178 + 8.4043*a*ρ_I/ρ_W - 0.0932*(Tavg-mt))
    c = firedrake.conditional(ρ<ρ_crit,a**ρ_I/ρ_W*β1*8.36*(mt-T)**-2.061,a**ρ_I/ρ_W*β2*8.36*(mt-T)**-2.061)
    dρdt=c*((ρ_I * year**2 / 1.0e-6) - ρ)
    
    return dρdt

def Li_2015(ρ, a, T, **kwargs):
    r"""
    """
    Tavg = utilities.lift3d(utilities.depth_average(T))
    β1 = -1.218 - 0.403*(Tavg-mt)
    β2 = β1*(0.792-1.080*a*ρ_I/ρ_W + 0.00465*(Tavg-mt))
    c = firedrake.conditional(ρ<ρ_crit,a*ρ_I/ρ_W*β1*8.36*(mt-T)**-2.061,a*ρ_I/ρ_W*β2*8.36*(mt-T)**-2.061)
    dρdt=c*((ρ_I * year**2 / 1.0e-6) - ρ)
    
    return dρdt

def Simonsen_2013(ρ, a, T, ρ_crit=550.0, ar1=0.07, ar2=0.03, F0=0.8, F1=1.25, **kwargs):
    r""" Simonsen 2013 paper
    """

    Tavg = utilities.lift3d(utilities.depth_average(T))
    gamma=61.7/((a*ρ_I/ρ_W)**.5)*exp(-3800/(R*Tavg))
    c = firedrake.conditional(ρ<ρ_crit,F0*ar1*a*ρ_I/ρ_W*g*exp(-Ec/(R*T)+Eg/(R*Tavg)),F1*ar2*a*ρ_I/ρ_W*g*exp(-Ec/(R*T)+Eg/(R*Tavg)))
    dρdt=c*((ρ_I * year**2 / 1.0e-6) - ρ)

    return dρdt


def Ligtenberg_2011(ρ, a, T, ρ_crit=550.0, ar1=0.07, ar2=0.03, **kwargs):
    r""" Ligtenberg densification model
    """

    m0 = max_value(1.435 - 0.151 * log(a*(ρ_I * year**2 / 1.0e-6)),firedrake.Constant(.25))
    m1 = max_value(2.366 - 0.293 * log(a*(ρ_I * year**2 / 1.0e-6)),firedrake.Constant(.25))
    Tavg = utilities.lift3d(utilities.depth_average(T))
    c = firedrake.conditional(ρ<ρ_crit,m0*ar1*a*(ρ_I * year**2 / 1.0e-6)*g*exp(-Ec/(R*T) + Eg/(R*Tavg)),m1*ar2*a*(ρ_I * year**2 / 1.0e-6)*g*exp(-Ec/(R*T) + Eg/(R*Tavg)))
    dρdt = c*((ρ_I * year**2 / 1.0e-6)-ρ)
    
    return dρdt


def Barnola_1991(ρ, a, T, ρ_crit=550.0, Q1=10160.0, k1=11.0,
    aHL=1.0, α=-37.455, β=99.743, δ=-95.027,γ=30.673, A0b=2.54e4,
    n=3.0, Q=60.0e3, close_off=800.0, **kwargs):
    # Annika, this is another one where I need help understanding what Max is doing
    #/ also what whether this is even a model worth including!

    dρdt = c*((ρ_I * year**2 / 1.0e-6)-ρ)

    return None


def Morris_HL_2014(ρ, a, T, ρ_crit=550.0, ar1=0.07, ar2=0.03, QMorris = 60.e3, kHL = 11.0, Estar = 10.16e3, **kwargs):
    
    Tavg = utilities.lift3d(utilities.depth_average(T))
    slope = -0.0009667915546575245 * QMorris /1.e3 + 0.001947860800510695
    intercept = 0.29455063899108685* QMorris /1.e3 - 2.652540535829697
    δE = (slope*self.Tavg + intercept) * 1000
    kMoris = kHL * exp(-(Estar - δE) / (R * Tavg))
    # Annika, what do we need for each of these models?

    c = firedrake.conditional(ρ<ρ_crit,kMorris/(ρ_W* year**2 / 1.0e-6/1000.0*g)*(1/Hx),0.0)


    dρdt = c*((ρ_I * year**2 / 1.0e-6)-ρ)

    return None

def KuipersMunneke_2015(ρ, a, T, ρ_crit=550.0, ar1=0.07, ar2=0.03, **kwargs):

    m0 = max_value(1.042 - 0.0916 * log(a*(ρ_I * year**2 / 1.0e-6)),firedrake.Constant(.25))
    m1 = max_value(1.734 - 0.2039 * log(a*(ρ_I * year**2 / 1.0e-6)),firedrake.Constant(.25))

    c = firedrake.conditional(ρ<ρ_crit,m0*ar1*a*ρ_I/ρ_W*g*exp(-Ec/(R*T) + Eg/(R*Tavg)),m1*ar2*a*ρ_I/ρ_W*g*exp(-Ec/(R*T) + Eg/(R*Tavg)))

    dρdt = c*((ρ_I * year**2 / 1.0e-6)-ρ)

    return dρdt  


def Goujon_2003(ρ, a, T, ρ_crit=550.0, ar1=0.07, ar2=0.03, F0=0.8, F1=1.25, **kwargs):
    # Annika, this is another model where I could use some help with the development.
    return None


class FirnModel:
    r"""Class for modeling depth dependent density profile
        
    This class solves for the firn density and vertical velocity (additional 
    energy density solutions are also in the developmental scope of the model).
    This will allow a thermodynamically consistent derivation of the firn density.
    Alternative models for the densification rate can be supplied via dρdt.
    """

    def __init__(self, dρdt=HerronLangway):
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
        
        return w*-h_f*ρ.dx(2)/-h_f*ϕ*-h_f*dx

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
        dρdt = self.dρdt(ρ, a, T, **kwargs)
            
        Q = ρ.function_space()
        ϕ = firedrake.TestFunction(Q)
            
        return dρdt*ϕ*-h_f*dx

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


        return -ρ*w.dx(2)*(-h_f)/(-h_f)*η*-h_f*dx

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

        keys = ('firn_thickness','firn_velocity','density','temperature','accumulation')
        keys_alt = ('h_f','w_f','ρ','T_f','a')
        h_f, w, ρ, T, a = get_kwargs_alt(kwargs, keys, keys_alt)
        dρdt = self.dρdt(ρ, a, T, **kwargs)
            
        Q = w.function_space()
        η = firedrake.TestFunction(Q)
            
        return dρdt*η*-h_f*dx