#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from Ansatz import TwoLocalUniversal, SWAP_test
from Utils import store, load, rmse
from ChemInstance import Instance

from qiskit import QuantumCircuit, transpile
from scipy.optimize import minimize
from qiskit.quantum_info import partial_trace, state_fidelity, Statevector
import threading


import numpy as np
import random
import yappi
class Encoder:
    def __init__(
        self,
        base=0,
        target=0,
        instance=None,
        file="",
        ansatz=TwoLocalUniversal,
        repetitions=1,
        test=True
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

        if file:
            self.file = file
            self.base = int(file[-8])
            self.target = int(file[-6])
            self._from_file=True

        if base:
            self.base = base
        elif instance:
            self.instance = instance
            self.base = self.instance.num_qubits
            self.elements = self.instance.elements
            self.vqe_reference = "energies_{}_{}.json".format(*self.elements)
        if not file and not base and not instance:
            raise Exception("No number of starting qubits known.")

        if not file:
            self.file = "../data/qae_config_{}_{}.json".format(self.base, self.target)
        self.ansatz, self.theta = ansatz(self.base, repetitions=repetitions)
        self.num_params = self.ansatz.num_parameters
        #self.backend = AerSimulator(method="statevector")
        
        if file:
            self._load_config()


    def _load_config(self):
        filedata = load(self.file)
        #index = filedata["error"].index(
        #    min(filedata["error"])
        #)
        index = -1
        self.file_param = filedata["parameters"][index]
        self._current_error = filedata["error"][index]
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
                
        self.ref_state = Statevector(QuantumCircuit((self.base-self.target)))


    def _set_evolution(self):
        qubits = self.base+(self.base-self.target)
        circ = QuantumCircuit(qubits, 1)
        circ.append(self.ansatz, range(self.base))
        for i in range(self.base-self.target):
            circ.swap(self.base-i-1, circ.num_qubits-i-1)
        self.evolution = circ

    def _set_states(self):
        qubits = self.base+(self.base-self.target)
        self.states = []
        for state in self.init_states:
            circ = QuantumCircuit(qubits, 1)
            circ.append(state, range(self.base))

            
            state = Statevector(circ)
            self.states.append(state)
        
        self.ref_state = Statevector(QuantumCircuit((self.base-self.target)))


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
        #circ = self.circuits[i]
        i_state = self.states[i]
        #bound_circ = circ.assign_parameters(
        #    {self.theta: parameters}
        #)   
        bound_evolve = self.evolution.assign_parameters(
            {self.theta: parameters}
        )   
        new_state = i_state.evolve(bound_evolve)
        #st_vec = Statevector(bound_circ)
        state = partial_trace(new_state, range(0, self.base))
        fidelity = state_fidelity(self.ref_state, state)
        acc=1-fidelity
        return acc

    """
    def _run_encoder_SWAP(self, parameters):
        self.n = self.base
        self.m = self.target
        self.backend = AerSimulator(method="statevector")
        
        qubits = self.n + (self.n - self.m) + 1
        circ = QuantumCircuit(qubits, 1)
        circ.append(self.init_state, range(self.n))

        self.bound_encoder = self.ansatz.assign_parameters({self.theta: parameters})

        circ.append(self.bound_encoder, range(self.n))

        circ.append(
            SWAP_test(self.n, self.m),
            list(range(self.m, self.n))
            + list(range(qubits - 1 - (self.n - self.m), qubits - 1))
            + [qubits - 1],
            [0],
        )
        circ.save_state()
        qc_compiled = transpile(circ, self.backend)
        res = self.backend.run(qc_compiled)
        res_obj = res.result()
        counts = res_obj.get_statevector(circ).probabilities_dict(qargs=[circ.num_qubits-1])

        if "1" in counts:
            acc = counts["1"]
        else:
            acc = 0
        return acc
        """
    """
    def _costfun(self, parameters, points):
        inter_res = []
        for i in range(len(points)):
            res = self._run_encoder(parameters, i)
            inter_res.append(res)
        total_err = rmse(inter_res)
        print(total_err, self._current_error)
        if total_err < self._current_error:
            self._current_error = total_err
            self.best_param = parameters

        return total_err
    """
    def _costfun(self, parameters, points):
        inter_res = []
        threads = []
        def target_fun(i):
            res = self._run_encoder(parameters, i)
            inter_res.append(res)
        
        for i in range(len(points)):
            t = threading.Thread(target=target_fun, args=(i,))
            threads.append(t)
            
        for t in threads:
            t.start()
            
        for t in threads:
            t.join()
            
        total_err = rmse(inter_res)
        print(total_err, self._current_error)
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
        """
        for i in points:
            init = []
            if i in saved_points:
                print("reference param found")
                index = saved_points.index(i)
                init = loaded_data["UCCSD_parameters"][index]
            print("({}/{})".format(points.index(i) + 1, len(points)))

            self.instance.run(i, init_parameters=init)
            self.init_states.append(self.instance.bound_vqe)

            if self.test:
                    self.H.append(self.instance.hamiltonian)
                    self.vqe_test.append(self.instance.vqe_result.x)
            vqe_data = {
                "UCCSD_points": i,
                "UCCSD_parameters": list(self.instance.parameters),
                "UCCSD": self.instance.vqe_result.x,
            }
            if not loaded_data:
                store(self.vqe_reference, vqe_data)
        """
        def _targ_vqe(i):
            init = []
            print("({}/{}):{}".format(points.index(i) + 1, len(points), i))
            if i in saved_points:
                print("reference param found")
                index = saved_points.index(i)
                init = loaded_data["UCCSD_parameters"][index]
            else:
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
        
        print("!!", self.init_states)

    def train(
        self,
        threshold=1e-8,
        num_states=1,
        max_iter=100,
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
            points = [1]
        else:
            points = list(np.arange(0.2, 3.2, 2.8 / (self.num_states-1)))
        self._generate_VQE_states(points)
        #self._set_circuits()
        self._set_states()
        self._set_evolution()
        print(threshold)
        yappi.start()
        
        #while self._current_error > threshold:
        if self._from_file:
            parameters = self.file_param
        else:
            parameters = [random.random() * np.pi for i in range(self.num_params)]
            
        if not self.test:
            res = minimize(
                self._costfun,
                parameters,
                args=(points),
                options={"maxiter": max_iter},
                tol=1e-2,
                method="SLSQP"
               # parallel={"max_workers":1}
            )
            print(res)

        else:
            for i in range(1):
                print(self.states)
                if self._from_file:
                    parameters = self.file_param
                else:
                    parameters = [random.random() * np.pi for i in range(self.num_params)]
                    
                    
                res = self._costfun(parameters, points)
    
                print(res)
        
        
        
        yappi.stop()
        threads = yappi.get_thread_stats()
        for thread in threads:
            print(
                "Function stats for (%s) (%d)" % (thread.name, thread.id)
            )  # it is the Thread.__class__.__name__
            yappi.get_func_stats(ctx_id=thread.id).print_all()


        config = {"parameters": [list(self.best_param)], "error": [self._current_error]}
        store(self.file, config)
        
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
        return self._current_error
        
