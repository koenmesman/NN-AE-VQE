#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import (
    JordanWignerMapper,
    BravyiKitaevSuperFastMapper,
    ParityMapper,
)
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SLSQP
from scipy.optimize import minimize
from qiskit.circuit.library import EfficientSU2
from qiskit_aer.primitives import Estimator
import numpy as np
import random
from optimparallel import minimize_parallel

from qiskit_algorithms.utils import algorithm_globals
algorithm_globals.random_seed = 512

print("source: examples")


def run_pub(param, circ, obs, estimator):
    pub = (circ, obs, param)
    job = estimator.run([pub])
    vqe_result = job.result()[0].data.evs
    return vqe_result


class Instance:
    def __init__(self, elements, mapping="JW", method="UCCSD", ansatz=None, freeze=[]):
        self.elements = elements
        self.atoms = "{} 0 0 0; {} 0 0 {}".format(*self.elements, 1)

        self.freeze = freeze
        self.method = method
        self.ansatz=ansatz
        self._set_driver(self.atoms)

        self._num_qubits = None
        self._set_mapper(mapping)
        self.est = Estimator
        self.backend_flag = False        

    def _set_driver(self, atoms):
        self._driver = PySCFDriver(
            atom=atoms,
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        )
        if self.freeze:
            properties = self._driver.run()
            self.problem = FreezeCoreTransformer(
                freeze_core=True, remove_orbitals=self.freeze
            ).transform(properties)
        else:
            self.problem = self._driver.run()

    @property
    def num_qubits(self):
        if self._num_qubits is None:

            # TODO: this should be more efficient

            #atoms = "{} 0 0 0; {} 0 0 {}".format(*self.elements, 1)
            #self._set_driver(atoms)

            self.ansatz = UCCSD(
                self.problem.num_spatial_orbitals,
                self.problem.num_particles,
                self.mapper,
                initial_state=HartreeFock(
                    self.problem.num_spatial_orbitals,
                    self.problem.num_particles,
                    self.mapper,
                ),
            )
            self._num_qubits = self.ansatz.num_qubits
        return self._num_qubits

    def set_hamiltonian(self, point):
        atoms = "{} 0 0 0; {} 0 0 {}".format(*self.elements, point)
        self._set_driver(atoms)
        second_q_op = self.problem.hamiltonian.second_q_op()
        self.hamiltonian = self.mapper.map(second_q_op)
        self.repulsion = self.problem.nuclear_repulsion_energy

    def _set_mapper(self, mapping):
        self.mapper = {
            "JW": JordanWignerMapper(),
            "BK": BravyiKitaevSuperFastMapper(),
            "parity": ParityMapper(self.problem.num_particles),
        }[mapping]

    def run(self, point, init_parameters=[], parallel=1):
        atoms = "{} 0 0 0; {} 0 0 {}".format(*self.elements, point)
        self._set_driver(atoms)
        #self.repulsion = self.problem.nuclear_repulsion_energy
        self.parallel = parallel
        # Hamiltonian setup
        second_q_op = self.problem.hamiltonian.second_q_op()
        self.hamiltonian = self.mapper.map(second_q_op)
        #self.H = [i for i in second_q_op.values()]

        # hamiltonian = self.problem.hamiltonian
        # self.init_H = hamiltonian
        # self.num_particles = self.problem.num_particles

        # case:
        # UCCSD : run UCCSD
        # SU2 : run SU2
        # exact : run exact
        result = {
            "UCCSD": self.run_VQE_UCCSD,
            "SU2": self.run_VQE_SU2,
            "exact": self.run_exact,
            "custom": self.run_custom_ansatz,
        }.get(self.method)(init_parameters)
        
        return result

    def runVQEV2(self, init_parameters):
        if self.backend_flag:
            estimator = self.est(backend_options={"max_parallel_experiments":0, "blocking_enable":True, 
                                               "blocking_qubits":2, "batched_shots_gpu":True,
                                               "runtime_parameter_bind_enable":True})
        else:
            estimator = self.est()
        self.vqe = VQE(ansatz=self.ansatz, estimator=estimator, optimizer=SLSQP())
        
        if not init_parameters:
            init_parameters = [
                random.random() * np.pi for i in range(self.ansatz.num_parameters)
            ]


        
        #opt = SLSQP()
        if self.parallel==1:
            self.vqe_result = minimize_parallel(fun=run_pub, x0=init_parameters, 
                                       bounds=tuple([(0, 2*np.pi)]*len(init_parameters)),
                                       args=[self.ansatz, self.hamiltonian, estimator],
                                       parallel={"loginfo": True})
        else:
            self.vqe_result = minimize(fun=run_pub, x0=init_parameters, 
                                       bounds=tuple([(0, 2*np.pi)]*len(init_parameters)),
                                       args=(self.ansatz, self.hamiltonian, estimator),
                                       method="SLSQP")
        #print(self.vqe_result)
        #print(self.vqe_result.loginfo)
        self.parameters = self.vqe_result.x

        self.bound_vqe = self.ansatz.assign_parameters(self.parameters)


    def runVQE(self, init_parameters):
        if self.backend_flag:
            estimator = self.est(backend_options={"max_parallel_experiments":0, "blocking_enable":True, 
                                               "blocking_qubits":2, "batched_shots_gpu":True,
                                               "runtime_parameter_bind_enable":True})
        else:
            estimator = self.est()
        self.vqe = VQE(ansatz=self.ansatz, estimator=estimator, optimizer=SLSQP())

        if not init_parameters:
            init_parameters = [
                random.random() * np.pi for i in range(self.ansatz.num_parameters)
            ]

        self.vqe.initial_point = init_parameters
        self.vqe_result = self.vqe.compute_minimum_eigenvalue(self.hamiltonian)
        self.parameters = self.vqe_result.optimal_parameters.values()
        self.bound_vqe = self.ansatz.assign_parameters(self.parameters)



    def run_VQE_UCCSD(self, init_parameters):
        self.ansatz = UCCSD(
            self.problem.num_spatial_orbitals,
            self.problem.num_particles,
            self.mapper,
            initial_state=HartreeFock(
                self.problem.num_spatial_orbitals,
                self.problem.num_particles,
                self.mapper,
            ),
        )
        if self.backend_flag:
            self.runVQE(init_parameters)
        else:
            self.runVQEV2(init_parameters)
        self._result = self.vqe_result
        try:
            return self._result.optimal_value
        except:
            return self.vqe_result

    def run_VQE_SU2(self, init_parameters):
        self.ansatz = EfficientSU2(self.hamiltonian.num_qubits)
        self.runVQE(init_parameters)
        self._result = self.vqe_result
        return self._result.optimal_value


    def run_custom_ansatz(self, init_parameters):
        self.runVQE(init_parameters)
        self._result = self.vqe_result
        return self._result.optimal_value


    def run_exact(self, init_parameters):
        numpy_solver = NumPyMinimumEigensolver()
        calc = GroundStateEigensolver(self.mapper, numpy_solver)
        self._result = calc.solve(self.problem)
        print(self._result._computed_energies[0])
        return self._result._computed_energies[0]
    
    def assign_parameters(self, point, parameters):
        atoms = "{} 0 0 0; {} 0 0 {}".format(*self.elements, point)
        self._set_driver(atoms)
        second_q_op = self.problem.hamiltonian.second_q_op()
        self.hamiltonian = self.mapper.map(second_q_op)
        self. parameters = parameters

        self.bound_vqe = self.ansatz.assign_parameters(parameters)


        
