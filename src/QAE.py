from Utils import load, rmse
from VQEBase import *

from qiskit import QuantumCircuit
from qiskit.quantum_info import partial_trace, state_fidelity, Statevector
from scipy.optimize import minimize
from math import floor
import random

class QAE():
    def __init__(self, ansatz:QuantumCircuit, base:int, target:int):
        """
        ansatz: variational quantum circuit
        base: number of qubits in full statespace
        target: number of qubits for encoded space
        """

        self.ansatz = ansatz
        self.base = base
        self.target = target

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

    def _set_reference_states(self, reference_data, reference_type, num_samples, sample_method):
        data = load(reference_data)[reference_type]

        points = data['points']
        parameters = data['parameters']

        if sample_method == "linear":
            indices = [floor(a) for a in np.linspace(0, len(points)-1, num_samples)]
        if sample_method == "random":
            indices = random.shuffle(range(0, len(points)))[0:num_samples]
        sample_points = [points[s] for s in indices]
        sample_parameters = [parameters[s] for s in indices]

        self.init_states = []

        for i in range(len(sample_points)):
            vqe = VQEExtended()
            circ = vqe.assign_parameters(sample_points[i], sample_parameters[i])
            self.init_states.append(circ)

    def _set_states(self):
        qubits = self.base+(self.base-self.target)
        self.states = []
        for state in self.init_states:
            circ = QuantumCircuit(qubits, 1)
            circ.append(state, range(self.base))

            
            state = Statevector(circ)
            self.states.append(state)
        
        self.ref_state = Statevector(QuantumCircuit((self.base-self.target)))


    def _set_evolution(self):
        qubits = self.base+(self.base-self.target)
        circ = QuantumCircuit(qubits, 1)
        circ.append(self.ansatz, range(self.base))
        for i in range(self.base-self.target):
            circ.swap(self.base-i-1, circ.num_qubits-i-1)
        self.evolution = circ

    def _init_parameters(self):
        """Generate random initial parameters."""
        return [random.random() * np.pi for _ in range(self.ansatz.num_parameters)]

    def _cost_trace(self, parameters):
        results=[]
        for state in self.states:
            bound_evolve = self.evolution.assign_parameters(parameters)   
            new_state = state.evolve(bound_evolve)
            state = partial_trace(new_state, range(0, self.base))
            fidelity = state_fidelity(self.ref_state, state)
            results.append(1-fidelity)
        return rmse(results)

    def train(self, reference_data:str, reference_type:str="VQE-UCCSD", num_samples = 2,
     sample_method='linear', initial_parameters=None, method:str="trace",
     optimizer:str="SLSQP", tol=1e-2, max_iter=1000):

        self._set_reference_states(reference_data, reference_type, num_samples, sample_method)
        self._set_states()
        self._set_evolution()
        
        initial_parameters = self._init_parameters() if initial_parameters is None else initial_parameters

        if method=="trace":
            costfun = self._cost_trace
        else:
            raise NotImplementedError("This method is not (yet) implemented.")
        
        res = minimize(
                costfun,
                initial_parameters,
                options={"maxiter": max_iter},
                tol=tol,
                method=optimizer
            )

        """
        data is structured as: {<compression>: {accuracy:double,
        'parameters':list(double), 'samples':int 'ansatz':str}}
        """
        data = {"{}_{}".format(self.base, self.target):{'accuracy':res.fun, 'parameters':list(res.x),
         'samples':num_samples, 'ansatz':self.ansatz.name}}

        return data
        