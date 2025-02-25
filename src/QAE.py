#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from Ansatz import TwoLocalUniversal, SWAP_test
from Utils import store, load, rmse
from ChemInstance import Instance

from qiskit import QuantumCircuit, transpile
from scipy.optimize import minimize, basinhopping
from qiskit.quantum_info import partial_trace, state_fidelity, Statevector
import threading
from multiprocessing import Process, Queue

from qiskit_aer.quantum_info import AerStatevector
#from qiskit_aer import StatevectorSimulator
import time
import numpy as np
import random
import os.path
import cma
#import yappi

class Encoder:
    def __init__(
        self,
        base=0,
        target=0,
        instance=None,
        file="",
        ansatz=TwoLocalUniversal,
        repetitions=1,
        test=False
    ):
        if not target and not file:
            raise Exception("No number of target qubits given.")
        self.target = target
        self._current_error=1
        self._from_file=False
        self.vqe_reference = ""
        self.test = test
        self.H = []
        self.vqe_test = []
        self.prog = []
        self.file = file

        if os.path.isfile(file):
            self.base = int(file[-8])
            self.target = int(file[-6])
            self._from_file=True

        if base:
            self.base = base
        elif instance:
            self.instance = instance
            self.base = self.instance.num_qubits
            self.elements = self.instance.elements
            self.vqe_reference = "../data/energies_{}_{}.json".format(*self.elements)
        if not file and not base and not instance:
            raise Exception("No number of starting qubits known.")

        if not file:
            self.file = "../data/qae_config_{}_{}.json".format(self.base, self.target)
        self.ansatz, self.theta = ansatz(self.base, repetitions=repetitions)

        self.num_params = self.ansatz.num_parameters
        #self.backend = AerSimulator(method="statevector")
        self.aer=False
        if os.path.isfile(file):
            self._load_config()


    def _load_config(self):
        filedata = load(self.file)
        index = filedata["error"].index(
            min(filedata["error"])
        )
        #index = -1
        self.file_param = filedata["parameters"][index]
        self._current_error = filedata["error"][index]
        print(self.ansatz.num_parameters)
        self.bound_encoder = self.ansatz.assign_parameters(
            {self.theta: self.file_param}
        )


    def _set_circuits(self):
        qubits = self.base+(self.base-self.target)
        self.circuits = []
        for state in self.init_states:
            circ = QuantumCircuit(qubits, 1)
            circ.append(state, range(self.base))
            circ.append(self.ansatz, range(self.base))
            for i in range(self.base-self.target):
                circ.swap(self.base-i-1, circ.num_qubits-i-1)
            
            
            self.circuits.append(circ)
                
        self.ref_state = AerStatevector(QuantumCircuit((self.base-self.target)))


    def _set_evolution(self):
        qubits = self.base+(self.base-self.target)
        circ = QuantumCircuit(qubits, 1)
        circ.append(self.ansatz, range(self.base))
        for i in range(self.base-self.target):
            circ.swap(self.base-i-1, circ.num_qubits-i-1)
            
            
            
        self.evolution = circ


    def _set_states(self):
        start = time.time()
        qubits = self.base+(self.base-self.target)
        self.states = []
        for s in self.init_states:      #Todo: multithread, should only take 50 secs
            circ = QuantumCircuit(qubits, 1)
            circ.append(s, range(self.base))
        
            if self.aer:
                state = AerStatevector(circ, device='gpu')
            else:
                state = AerStatevector(circ)

            self.states.append(state)
        
        self.ref_state = AerStatevector(QuantumCircuit((self.base-self.target)))
        if self.aer:
            self.ref_state = AerStatevector(self.ref_state, device='gpu')
        print("states set {}".format(time.time()-start))
    
    def _run_encoder(self, parameters, i):
        """
        init_state = self.init_states[i]

        qubits = self.base+(self.base-self.target)
        
        ref_state = Statevector(QuantumCircuit((self.base-self.target)))
        
        
        circ = QuantumCircuit(qubits, 1)
        circ.append(init_state, range(self.base))
        bound_encoder = self.ansatz.assign_parameters(
            {self.theta: parameters}
        )
        circ.append(bound_encoder, range(self.base))
        for i in range(self.base-self.target):
            circ.swap(self.base-i-1, circ.num_qubits-i-1)
        st_vec = Statevector(circ)
        state = partial_trace(st_vec, range(0, self.base))
        fidelity = state_fidelity(ref_state, state)
        acc=1-fidelity
        print(acc)
        """
        
        #print(self.states)
        start = time.time()
        #circ = self.circuits[i]
        i_state = self.states[i]
        #bound_circ = circ.assign_parameters(
        #    {self.theta: parameters}
        #)   
        bound_evolve = self.evolution.assign_parameters(
            {self.theta: parameters}
        )   
        #print("parameters assigned {}".format(time.time()-start))
        new_state = i_state.evolve(bound_evolve)
        
        
        #new_state = new_state.evolve(self.evolution[1])
        #print("state evolved {}".format(time.time()-start))
        #st_vec = Statevector(bound_circ)
        state = partial_trace(new_state, range(0, self.base))
        #print("partial trace {}".format(time.time()-start))
        fidelity = state_fidelity(self.ref_state, state)
        #print("fidelity calc {}".format(time.time()-start))
        acc=1-fidelity
        #print("accuracy:", acc)
        
        return acc

    def _costfun(self, parameters, points):
        inter_res = []
        threads = []
        def target_fun(i):
            res = self._run_encoder(parameters, i)
            #print(res)
            inter_res.append(res)
            return res

        def f(q, parameters, i):
            res = self._run_encoder(parameters, i)
            q.put(res)
            
        q = Queue()
        for i in range(len(points)):
            t = Process(target=f, args=(q, parameters, i))
            threads.append(t)
            
        for t in threads:
            t.start()
            
        for t in threads:
            t.join()
        
        inter_res = q.get()

        

        total_err = rmse(inter_res)
        print(total_err, self._current_error)
        self.prog.append(total_err)
        if total_err < self._current_error:
            self._current_error = total_err
            self.best_param = parameters

        return total_err
    
    def _generate_VQE_states(self, points):
        self.init_states = []

        loaded_data = load(self.vqe_reference)
        if not loaded_data:
            loaded_data={"points":[]}
        saved_points = loaded_data["UCCSD_points"]
        print("finding initial reference points")

        def _targ_vqe(i):
            init = []
            print("({}/{}):{}".format(points.index(i) + 1, len(points), i))
            if i in saved_points:
                print("reference param found")
                index = saved_points.index(i)
                init = loaded_data["UCCSD_parameters"][index]
                print(i, len(loaded_data["UCCSD_parameters"]))
            else:
                print("not found: ", i)
                self.instance.run(i, init_parameters=init)
            self.instance.assign_parameters(i, init)
            self.init_states.append(self.instance.bound_vqe)

            if self.test:
                    self.H.append(self.instance.hamiltonian)
                    #self.vqe_test.append(self.instance.vqe_result.x)

            if not loaded_data:
                vqe_data = {
                    "UCCSD_points": i,
                    "UCCSD_parameters": list(self.instance.parameters),
                    "UCCSD": self.instance.vqe_result.x,
                }
                store(self.vqe_reference, vqe_data)
        
        threads=[]
        for i in points:
            t = threading.Thread(target=_targ_vqe, args=(i,))
            threads.append(t)
            
        for t in threads:
            t.start()
            
        for t in threads:
            t.join()
        print("finished loading state data")
        
    def train(
        self,
        threshold=1e-8,
        num_states=1,
        max_iter=10000,
        mapping="JW",
        freeze=[],
        atom=[],
        vqe_reference="",
    ):
        if vqe_reference:
            self.vqe_reference = vqe_reference
        elif atom:
            self.vqe_reference = "../data/Energies_{}_{}.json".format(*self.elements)
        else:
            raise Exception("No atom elements known.")
        if not self.instance:
            if not atom:
                raise Exception("No atom elements known.")
            self.atom = atom
            self.instance = Instance(atom)
            self.instance.freeze = freeze
            self.instance.mapping = mapping
        self.num_states = num_states

        if self._from_file:
            parameters = self.file_param
            self.best_param = parameters

        else:
            parameters = [random.random() * np.pi for i in range(self.num_params)]
            self._current_error = 1
            self.best_param = []

        if self.num_states==1:
            points = [0.2]
        else:
            points = list(np.arange(0.2, 3.2, 2.8 / (self.num_states-1)))
        self._generate_VQE_states(points)
        #self._set_circuits()
        self._set_states()
        self._set_evolution()
        #yappi.start()
        
        #while self._current_error > threshold:
        if self._from_file:
            parameters = self.file_param
        else:
            parameters = [random.random() * np.pi for i in range(self.num_params)]
            
        bounds = [(0, 2*np.pi)]*self.num_params
            
        if not self.test:   
            """
            res = basinhopping(
                self._costfun,
                parameters,
                #bounds=bounds,
                niter=5,
                T=0.4,
                stepsize=1,
                stepwise_factor=0.6,
                interval=10000,
                minimizer_kwargs={
                "method":"COBYLA",
                "args":(points),
                "options":{"maxiter": max_iter},
                "tol":1e-8,
                }
               # parallel={"max_workers":1}
            )
            """
            """
            res = minimize(
                self._costfun,
                parameters,
                bounds=bounds,
                method = "COBYLA",
                args = (points),
                options = {"maxiter": max_iter},
                tol = 1e-8,
                
               # parallel={"max_workers":1}
            )
            """
            bounds = [[0]*len(parameters), [2*np.pi]*len(parameters)]
            print(points)
            res, es = cma.fmin2(self._costfun, parameters, 0.5, options={'bounds':bounds}, args=([points]))
            print(res)
            
        else:
            for i in range(10):
                if self._from_file:
                    parameters = self.file_param
                else:
                    parameters = [random.random() * np.pi for i in range(self.num_params)]
                    
                    
                res = self._costfun(parameters, points)
    
                print(res)
        
        
        """
        #yappi.stop()
        #threads = yappi.get_thread_stats()
        for thread in threads:
            print(
                "Function stats for (%s) (%d)" % (thread.name, thread.id)
            )  # it is the Thread.__class__.__name__
            yappi.get_func_stats(ctx_id=thread.id).print_all()
        """

        config = {"parameters": [list(self.best_param)], "error": [self._current_error]}
        store(self.file, config)
        """
        if self.test:
            from qiskit.primitives import Estimator
            estimator = Estimator()
            self._load_config()
            enc_qc = self.bound_encoder
            dec_qc = enc_qc.inverse()
            E_QAE = []
            for vqe_state, H in zip(self.init_states, self.H): 
                
                qc = QuantumCircuit(self.base)
                qc.append(vqe_state, range(self.base)) # VQE
                qc.append(enc_qc, range(self.base))
                qc.reset(3)
                qc.append(dec_qc, range(self.base))
                ansatz=qc
                
                #E_repulsion = inst.problem.nuclear_repulsion_energy
                est = estimator.run([ansatz], [H]).result()

                E_QAE.append(est.values[0])
            print(E_QAE)
        """
        return self._current_error
    
    
    def print_prog(self):
        import matplotlib.pyplot as plt
        plt.plot(range(len(self.prog)), self.prog)
        plt.savefig("optimization_QAE.png", format='png', bbox_inches='tight')

        
        
        
    