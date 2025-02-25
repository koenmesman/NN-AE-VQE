#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:09:14 2025

@author: kmesman
"""
from Ansatz import StronglyEntangled, TwoLocalU3
from Utils import store, load
from QAE import Encoder

from qiskit.primitives import Estimator, StatevectorEstimator
from qiskit import QuantumCircuit
from scipy.optimize import minimize, basinhopping
from os.path import exists
import numpy as np
import random
import json
import copy
import cma

class VQEResult():
    def __init__(self):
        self.parameters = []
        self.point = []
        self.energy = []
        self.repulsion_energy = []
        #self.H = []
        self.error = []
        
    def append(self, data):
        if type(data)==VQEResult:
            data = data.__dict__
        
        for k in data.keys():
            c = getattr(self, k)
            c.append(data[k])
            setattr(self, k, c)
        

class AEVQE():
    def __init__(self, cheminstance, encoderfile, reference, layers=1, optimizer="COBYLA"):
        """
        load encoder
        set ansatz
        combine ansatz + decoder
        set estimator
        set cheminstance for Hamiltonian (optional input)
        
        set stored data
        """
        self.CI = cheminstance
        self.reference = load(reference)
        self.opt = optimizer
        self.encoder = Encoder(file=encoderfile, repetitions=1)
        self.decoder_circ = self.encoder.bound_encoder.inverse()
        self.qubits = self.decoder_circ.num_qubits
        self.ansatz, self.theta = TwoLocalU3(self.encoder.target, layers=layers)
        self.decoded_ansatz = self._construct_ansatz()
        
        self.estimator = StatevectorEstimator()
        self.result = VQEResult()
        self.optimizer = "L-BFGS-B"
        return

    def _construct_ansatz(self):
        qc = QuantumCircuit(self.encoder.base)
        qc.append(self.encoder.bound_encoder, range(0, self.encoder.base))
        qc.append(self.ansatz, range(0, self.encoder.target))
        qc.append(self.decoder_circ, range(0, self.encoder.base))
        return qc

    def _update_best(self, res, par):
        if res < self.best_result[0]:
            self.best_result = [res, par]
            
    def _reconstruct(self, par1, par2, index): #Todo: mapping function
        j1 = 0
        j2 = 0
        parameters = []
        for i in range(len(par1) + len(par2)):
            if i in list(index):
                parameters.append(par1[j1])
                j1 += 1
            else:
                parameters.append(par2[j2])
                j2 += 1
        return parameters
    
    def _find_global(self, global_point, tol=1e-4, current_err=2):
        """
        !!! Will need rewriting if lower values does not mean better accuracy.
        """
        self.best_result = [2, []]
        self.CI.set_hamiltonian(global_point)
        num_par = self.ansatz.num_parameters
        bnds  = tuple([(0, 2*np.pi)]*num_par)  
        ref_index = self.reference["exact_points"].index(global_point)
        ref_E = self.reference['exact'][ref_index]
        vqeres = VQEResult()
        vqeres.point = global_point
        vqeres.repulsion_energy = self.CI.repulsion

        tmp = VQEResult()
        tmp.append(vqeres)
        parameters = self._global_ref_param 
        if not self.reference:
            self.current_acc = -1e3    
            #parameters = [random.random() * 2 * np.pi for i in range(num_par)]
        init_err = current_err
        
        while current_err>tol:
            """
            res = minimize(
                self._estimate,
                parameters,
                args=(self.decoded_ansatz, ref_E),
                method=self.opt,
                bounds=bnds,
                options={"maxiter": 2000},
                tol=1e-8,
            )
            E = res.fun
            tmp.energy = [E]
            tmp.parameters = [list(res.x)]
            err = E
            
            if err>=current_err:
                parameters = self.best_result[1]
                #parameters = [random.random() * 2 * np.pi for i in range(num_par)]
                print("restart")
            else:
                parameters = res.x
                print("continue")
            """
            """
            res = basinhopping(self._estimate,
            parameters,
            minimizer_kwargs = {
                'args':(self.decoded_ansatz, ref_E),
                'method':self.optimizer,
                'bounds':bnds,
                'tol':1e-8},
            T=0.4,
            niter=5,
            interval=10000,
            stepsize=2,
            stepwise_factor=0.6
            )
            
            
            E = res.fun
            tmp.energy = [E]
            tmp.parameters = [list(res.x)]
            err = E
            tmp.error = [err]
            if err<init_err and self.always_store:
                store(self.filename, tmp.__dict__)  
            print(err)

            current_err = err
            
            """
            parameters = [random.random() * 2 * np.pi for i in range(len(parameters))]
            bnds = [[0]*len(parameters), [2*np.pi]*len(parameters)]
            
            res, es = cma.fmin2(self._estimate, parameters, 0.1, args=(self.decoded_ansatz, ref_E),
                                options={'bounds':bnds})
            print(res)
            
            
            E = res.fbest
            tmp.energy = [E]
            tmp.parameters = [list(res.xbest)]
            err = E
            tmp.error = [err]
            if err<init_err and self.always_store:
                store(self.filename, tmp.__dict__)  
            print(err)

            current_err = err
            
        store(self.filename, tmp.__dict__)  

        return res.x

    def load_point(self, point):
        self.CI.set_hamiltonian(point)
        ref_index = self.reference["exact_points"].index(point)
        ref_E = self.reference['exact'][ref_index]
        return self.CI.hamiltonian, ref_E


    def _estimate(self, par, circ, ref):
        pub = (circ, self.CI.hamiltonian, par)
        est = self.estimator.run([pub])
        res = est.result()[0].data.evs
        self._tmp_E = res
        res = abs(res-ref)
        self._update_best(res, par)
        print(res, "b:", self.best_result[0])
        return res
    
    def sequential_pop(self, l, index):
        index = list(index)
        index.sort()
        index.reverse()
        for i in index:
            l.pop(i)
        return l
    
    def _get_global_ref(self):
        with open(self.filename, "r") as file:
            data = json.load(file)
            self.best_ref = min(data["error"])
            print(self.best_ref)
            index = data["error"].index(self.best_ref)
            ref = {}
            for k in data.keys():
                ref[k] = data[k][index]
        return ref
    
    def _get_point(self, parameters, p, ref_E):
        vqeres = VQEResult()
        vqeres.point = [p]
        self.CI.set_hamiltonian(p)

        vqeres.repulsion_energy = [self.CI.repulsion]
        err=2
        #parameters = [random.random() * 2 * np.pi for i in range(len(parameters))]

        while err>1e-5:
            res = minimize(
                self._estimate,
                parameters,
                args=(self.decoded_ansatz, ref_E),
                method=self.optimizer,
                bounds=self.bounds,
                options={"maxiter": 100},
                tol=1e-8,
            )
            print(res)
            err = res.fun
            vqeres.energy = [float(self._tmp_E)]
            vqeres.error = [err]
            vqeres.parameters = [list(res.x)]
            print("point {} found with accuracy {}".format(p, err))
        store(self.filename, vqeres.__dict__)
 
        
        return res.x
    
    def _find_points_gradient(self, points, rate=3):
        second_last = copy.deepcopy(self._global_ref_param)
        print("second", len(second_last))
        last_params = [random.random() * 2 * np.pi for i in range(self.ansatz.num_parameters)]

        bias = 0.3*np.pi/len(points)
        self.stop=False
        self.bounds = tuple([(0, 2*np.pi)]*len(last_params))
        i=0
        #counter=0
        
        for i in range(len(points)):
            p = points[i]
            #last_params = self.best_result[1]
            deltas = [b-a for a,b in zip(second_last, last_params)]
            second_last = copy.deepcopy(last_params)
            print(deltas)
            if i:
                self.bounds = [(x+delta-rate*(bias+abs(delta)), x+delta+rate*(bias+abs(delta))) for x, delta in zip(last_params, deltas)]
            print("atomic distace: {}, ({}/{})".format(p, i+1, len(points)))
            #self.vqe_res = self.es.run_exact(p)
            #self.hamiltonian = self.es.hamiltonian
            self.hamiltonian, E = self.load_point(p)

            #self._local_par = copy.deepcopy(self._global_ref_param)
            new_par = copy.deepcopy(self.best_result[1])
            last_params = self._get_point(new_par, p, E)
            #new_par = copy.deepcopy(self.best_result[1])
            #E, last_params = self._seq_minimize(new_par, p, E)
            # !!! needs fix, delta calculation not with right parameters, order wrong

    def _estimate_seq(self, par, par2, index, ref_E):
        parameters = self._reconstruct(par, par2, index)
        return self._estimate(parameters, self.decoded_ansatz, ref_E)

    def _seq_minimize(self, parameters, p, E):
        vqeres = VQEResult()
        vqeres.point = [p]
        self.CI.set_hamiltonian(p)

        vqeres.repulsion_energy = [self.CI.repulsion]
        
        
        # U1
        i1 = range(2, len(parameters), 3)
        i2 = range(0, len(parameters), 3)
        # parameters = [np.pi/2 if i in i2 else parameters[i] for i in range(len(parameters))]
        par1 = [parameters[x] for x in i1]
        par2 = self.sequential_pop(parameters, i1)
        

        
        bnds1 = tuple([self.bounds[x] for x in i1])
        acc = 1
        while acc>1e-5:
            res = minimize(
                self._estimate_seq,
                par1,
                #bounds=bnds1,
                args=(par2, i1, E),
                method=self.optimizer,
                options={"maxiter": 10000},
                tol=1e-8,
            )
            print(res.fun)
            acc = res.fun
        parameters = self._reconstruct(res.x, par2, i1)

        # U3 (theta)
        i2 = range(2, len(parameters), 3)
        i1 = self.sequential_pop(list(range(len(parameters))), i2)
        par2 = [parameters[x] for x in i2]
        par1 = self.sequential_pop(parameters, i2)
        bnds2 = tuple([self.bounds[x] for x in i1])
        res = minimize(
            self._estimate_seq,
            par1,
            #bounds=bnds2,
            args=(par2, i1, E),
            method=self.optimizer,
            options={"maxiter": 10000},
            tol=1e-8,
        )

        parameters = self._reconstruct(res.x, par2, i1)

        err = res.fun
        vqeres.energy = [float(self._tmp_E)]
        vqeres.error = [err]
        vqeres.parameters = [parameters]
        print("point {} found with accuracy {}".format(p, err))
        store(self.filename, vqeres.__dict__)


        return res.fun, parameters

    
    def find_parameters(self, samples, filename, global_err=1e-4,
                        global_point = None, locked_parameters={},
                        always_store=False):
        self.filename=filename
        self.always_store = always_store
        if always_store:
            self._c = 0
        # Configure initial paramters
        if exists(self.filename):
            self._global_ref = self._get_global_ref()
            self._global_ref_param = self._global_ref['parameters']
            current_err = self._global_ref['error']
            if not global_point:
                global_point = self._global_ref['point']
        else:
            self._global_ref_param = [random.random() * 2 * np.pi for i in range(self.ansatz.num_parameters)]
            current_err = 1
            if not global_point:
                global_point = samples[0]
        
        print("current err: {}".format(current_err))
        

        
        # search for point that satisfies error tolerance
        if current_err>=global_err:
            self._global_ref_param = self._find_global(global_point, tol=global_err,
                                             current_err=current_err)
        ref_par = copy.deepcopy(self._global_ref_param)
        self.best_result = [current_err, ref_par]

        self._find_points_gradient(samples)
        return
