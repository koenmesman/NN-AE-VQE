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

class Instance:
    def __init__(self, elements, mapping="JW", method="UCCSD", ansatz=None, freeze=[]):
        self.elements = elements
        self._set_mapper(mapping)
        self.freeze = freeze
        self.method = method
        self.ansatz=ansatz
        self._num_qubits = None
        self.est = Estimator
        self.backend_flag = True

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
        print(self.problem)

    def _set_mapper(self, mapping):
        self.mapper = {
            "JW": JordanWignerMapper(),
            "BK": BravyiKitaevSuperFastMapper(),
            "parity": ParityMapper(),
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
        result = {"exact": self.run_exact}.get(self.method)(init_parameters)
        
        return result

  
    def run_exact(self, init_parameters):
        numpy_solver = NumPyMinimumEigensolver()
        calc = GroundStateEigensolver(self.mapper, numpy_solver)
        print(self.problem)
        self._result = calc.solve(self.problem)
        print(self._result._computed_energies[0])
        return self._result._computed_energies[0]
