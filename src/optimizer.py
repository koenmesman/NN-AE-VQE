#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:20:05 2025

optimizers library
@author: kmesman
"""

from scipy.optimize import minimize, basinhopping
from optimparallel import minimize_parallel
import cma


class optimizer():
    def __init__(self, opt_name, secondary=None):
        self.method = opt_name
        self.secondary = secondary
        self.optim_fun = {
            "L-BFGS-B" : self.scipy_minimize,
            "Nelder-Mead" : self.scipy_minimize,
            "COBYLA" : self.scipy_minimize,
            "BasinHopping" : self.basin_hopping,
            "cma" : self.CMA,
            "parallel" : self.parallel
            }.get(opt_name)
    
    def opt(self, cost_fun, x0, args, options, bounds=[], tol=1e-8, **kwargs):
        return self.optim_fun(cost_fun, x0, args, options, bounds, tol, **kwargs)
    
    def scipy_minimize(self, cost_fun, x0, args, options, bounds, tol, **kwargs):
        res = minimize(
        cost_fun,
        x0,
        bounds=bounds,
        method = self.method,
        args = args,
        options = options
        )
        return [res.fun, res.x]
    
    def basin_hopping(self, cost_fun, x0, args, options, bounds, tol, **kwargs):
        res = basinhopping(
            cost_fun,
            x0,
            bounds=bounds,
            **kwargs,
            minimizer_kwargs={
            "method":self.secondary,
            "args":args,
            "options":options,
            "tol":tol,
            }
            )
        return [res.fun, res.x]
    
    def CMA(self, cost_fun, x0, args, options, bounds, tol, **kwargs):
        res, es = cma.fmin2(cost_fun, x0, **kwargs, args=args,
                            options=options)
        return [res.fbest, res.xbest]
    
    def parallel(self, cost_fun, x0, args, options, bounds, tol, **kwargs):
        res = minimize_parallel(
        cost_fun,
        x0,
        bounds=bounds,
        method = self.method,
        args = args,
        options = options
        )
        return [res.fun, res.x]