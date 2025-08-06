# -*- coding: utf-8 -*-
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit import QuantumCircuit
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms.optimizers import COBYLA
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
import random
import numpy as np
from qiskit_aer.primitives import Estimator
from qiskit.ibmruntime import IBMBackend

class AtomDriver(PySCFDriver):
    """
    A driver to set up a single atom (or multiple atoms) via PySCFDriver,
    designed as a subclass template under Qiskit Nature 2.1.1.
    """

    def __init__(
        self,
        atom: str = "H",
        charge: int = 0,
        spin: int = 0,
        basis: str = "sto3g",
        unit: str = DistanceUnit.ANGSTROM,
    ):
        """
        Build an AtomDriver for one or more atoms.

        Args:
            atom_symbol: e.g. "H", "He", or "O" etc. If multiple atoms, 
                         supply as string "H 0 0 0; O 0 0 1.0" etc.
            charge: integer molecular charge
            spin: total spin multiplicity (2S = multiplicity - 1)
            basis: basis set name, default "sto3g"
            unit: PySCF unit system (e.g. units.Angstrom or units.Bohr)
        """
        super().__init__(atom, charge=charge, spin=spin, basis=basis, unit=unit)


    def freeze_orbitals(self, orbitals=None):
        properties = self.run()
        self.problem = FreezeCoreTransformer(
                freeze_core=True, remove_orbitals=orbitals
            ).transform(properties)

    def run(self):
        """
        Executes calculation and returns ElectronicStructureDriverResult.
        Feel free to override this to inject pre/post processing.
        """
        result = super().run()
        print(result)

        self.problem = result
        # You can manipulate or annotate result here before returning
        return result


class VQEExtended():
    def __init__(self,
        ansatz:QuantumCircuit = None,
        mapper = JordanWignerMapper(),
        shots: int = 100,
        ):
        self.mapper = mapper
        self.ansatz = ansatz

    def _set_UCCSD(self, atom):
        ansatz = UCCSD(
            atom.problem.num_spatial_orbitals,
            atom.problem.num_particles,
            self.mapper,
            initial_state=HartreeFock(
                atom.problem.num_spatial_orbitals,
                atom.problem.num_particles,
                self.mapper,
            ),
        )

        return ansatz

    def _set_estimator(self, estimator, backend_options: dict = {}, optimizer=COBYLA()):
        self.vqe = VQE(ansatz=self.ansatz, estimator=estimator, optimizer=optimizer)

    def _set_backend(self, backend, backend_options: dict = {}, optimizer=COBYLA()):
        estimator = BackendEstimator(backend = backend)
        self.vqe = VQE(ansatz=self.ansatz, estimator=estimator, optimizer=optimizer)

    def _set_atom(self, atom_config):
        atom = AtomDriver(atom_config)
        atom.run()
        second_q_op = atom.problem.hamiltonian.second_q_op()
        if self.ansatz==None:
            self.ansatz=self._set_UCCSD(atom)
        
        self.num_qubits = self.ansatz.num_qubits
        self.hamiltonian = self.mapper.map(second_q_op)
        return atom

    def run_exact(self, atom_config: str = "H 0 0 0; H 0 0 0.5", init_parameters=None):
        atom = self._set_atom(atom_config)

        numpy_solver = NumPyMinimumEigensolver()
        calc = GroundStateEigensolver(self.mapper, numpy_solver)

        self._result = calc.solve(atom.problem)
        return self._result._computed_energies[0]

    def run(self, atom_config:str, backend = Estimator(), init_parameters=None, optimizer=COBYLA()):
        atom = self._set_atom(atom_config)
        
        #Todo check backend types
        if isinstance(backend, Estimator):
            self._set_estimator(backend, optimizer=optimizer)
        if isinstance(backend, IBMBackend):
                self._set_estimator(backend, optimizer=optimizer)


        if not init_parameters:
            init_parameters = [
                random.random() * np.pi for i in range(self.ansatz.num_parameters)
            ]
        self.vqe.initial_point = init_parameters
        vqe_result = self.vqe.compute_minimum_eigenvalue(self.hamiltonian)
        parameters = vqe_result.optimal_parameters.values()
        bound_vqe = self.vqe.ansatz.assign_parameters(parameters)

        return {"energies":vqe_result.optimal_value, "parameters":parameters, "ansatz":bound_vqe} #Todo validate value