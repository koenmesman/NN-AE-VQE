# -*- coding: utf-8 -*-
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import BaseEstimatorV2
from qiskit import QuantumCircuit
import random
import numpy as np


class AtomDriver(PySCFDriver):
    """
    A driver to set up a single atom (or multiple atoms) via PySCFDriver,
    designed as a subclass template under Qiskit Nature 1.4.3.
    """

    def __init__(self,
                 atom: str = "H",
                 charge: int = 0,
                 spin: int = 0,
                 basis: str = "sto3g",
                 unit: str = DistanceUnit.ANGSTROM):
        """
        Build an AtomDriver for one or more atoms.

        Args:
            atom: Molecule definition string (e.g., "H 0 0 0; O 0 0 1.0").
            charge: Integer molecular charge.
            spin: Total spin multiplicity (2S = multiplicity - 1).
            basis: Basis set name, default "sto3g".
            unit: Distance units (e.g., ANGSTROM or BOHR).
        """
        super().__init__(atom, charge=charge, spin=spin, basis=basis, unit=unit)

    def run(self):
        """
        Executes calculation and stores the problem.
        """
        result = super().run()
        self.problem = result
        return result


class VQEExtended:
    """
    Wrapper class for running VQE and exact diagonalization for molecular problems.
    """

    def __init__(self,
                 ansatz: QuantumCircuit = None,
                 mapper=JordanWignerMapper(),
                 shots: int = 1024):
        self.mapper = mapper
        self.ansatz = ansatz
        self.shots = shots
        self.vqe = None
        self.hamiltonian = None
        self.num_qubits = None

    # -------------------------------
    # Helper methods
    # -------------------------------

    def _build_uccsd_ansatz(self, atom):
        """Construct a default UCCSD ansatz."""
        return UCCSD(
            atom.problem.num_spatial_orbitals,
            atom.problem.num_particles,
            self.mapper,
            initial_state=HartreeFock(
                atom.problem.num_spatial_orbitals,
                atom.problem.num_particles,
                self.mapper,
            ),
        )

    def _set_atom(self, atom_config):
        """Prepare the molecular problem and ansatz."""
        atom = AtomDriver(atom_config)
        atom.run()
        second_q_op = atom.problem.hamiltonian.second_q_op()

        if self.ansatz is None:
            self.ansatz = self._build_uccsd_ansatz(atom)

        self.num_qubits = self.ansatz.num_qubits
        self.hamiltonian = self.mapper.map(second_q_op)
        return atom

    def _init_parameters(self):
        """Generate random initial parameters."""
        return [random.random() * np.pi for _ in range(self.ansatz.num_parameters)]

    # -------------------------------
    # Main execution methods
    # -------------------------------

    def run_exact(self, atom_config: str = "H 0 0 0; H 0 0 0.5"):
        """Perform exact diagonalization using a classical eigensolver."""
        atom = self._set_atom(atom_config)
        solver = NumPyMinimumEigensolver()
        calc = GroundStateEigensolver(self.mapper, solver)
        result = calc.solve(atom.problem)
        return result.groundenergy

    def run(self,
            atom_config: str,
            estimator: BaseEstimatorV2,
            backend_options: dict = None,
            init_parameters=None,
            optimizer=COBYLA()):
        """
        Run VQE with either a custom ansatz or default UCCSD.

        Args:
            atom_config: Molecule specification string.
            backend: Quantum backend (e.g., AerSimulator or IBMQ backend).
            backend_options: Optional dict for QuantumInstance settings.
            init_parameters: Optional list of initial ansatz parameters.
            optimizer: Classical optimizer (default: COBYLA).
        """
        self._set_atom(atom_config)

        # Setup VQE
        self.vqe = VQE(ansatz=self.ansatz, optimizer=optimizer, estimator=estimator)

        # Initialize parameters
        if not init_parameters:
            init_parameters = self._init_parameters()
        self.vqe.initial_point = init_parameters

        # Compute result
        vqe_result = self.vqe.compute_minimum_eigenvalue(self.hamiltonian)
        parameters = list(vqe_result.optimal_parameters.values())
        bound_vqe = self.vqe.ansatz.assign_parameters(parameters)

        return {
            "energy": vqe_result.optimal_value,
            "parameters": parameters
        }

    def assign_parameters(self, atom_config, parameters):
        self._set_atom(atom_config)

        if isinstance(self.ansatz, UCCSD):
            qc = QuantumCircuit(self.ansatz.num_qubits)
            qc.append(self.ansatz.initial_state, range(0, self.ansatz.num_qubits))
            qc.append(self.ansatz.assign_parameters(parameters), range(0, self.ansatz.num_qubits))
            return qc
        else:
            return self.ansatz.assign_parameters(parameters)
