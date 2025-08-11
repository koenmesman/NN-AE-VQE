# -*- coding: utf-8 -*-
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import BaseEstimatorV2, Estimator, StatevectorEstimator
from qiskit import QuantumCircuit
from multiprocessing import Pool
import random
import copy
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


def set_atom(config, mapper):
    """Prepare the molecular problem and ansatz."""
    atom = AtomDriver(config)
    atom.run()
    second_q_op = atom.problem.hamiltonian.second_q_op()
    hamiltonian = mapper.map(second_q_op)
    return atom, hamiltonian

def run_estimator(config):#:tuple[str, list[float]]):
    atom_config, parameters, ansatz, mapper = config
    atom, hamiltonian = set_atom(atom_config, mapper)
    estimator = StatevectorEstimator()
    #bound_circ = ansatz.assign_parameters(parameters)
    #print("!", bound_circ)

    pub = (ansatz, hamiltonian, parameters)
    est = estimator.run([pub])
    return est.result()[0].data.evs

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
        self.atom = AtomDriver(atom_config)
        self.atom.run()
        second_q_op = self.atom.problem.hamiltonian.second_q_op()

        if self.ansatz is None:
            self.ansatz = self._build_uccsd_ansatz(self.atom)

        self.num_qubits = self.ansatz.num_qubits
        self.hamiltonian = self.mapper.map(second_q_op)
        return self.atom, self.hamiltonian, self.ansatz

    def _init_parameters(self):
        """Generate random initial parameters."""
        return [random.random() * np.pi for _ in range(self.ansatz.num_parameters)]

    # -------------------------------
    # Main execution methods
    # -------------------------------

    def run_exact(self, atom_config: str = "H 0 0 0; H 0 0 0.5"):
        """Perform exact diagonalization using a classical eigensolver."""
        self._set_atom(atom_config)
        solver = NumPyMinimumEigensolver()
        calc = GroundStateEigensolver(self.mapper, solver)
        result = calc.solve(self.atom.problem)
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
        self.bound_vqe = self.vqe.ansatz.assign_parameters(parameters)

        return {
            "energy": vqe_result.optimal_value,
            "parameters": parameters
        }

    def _single_vqe(self, atom_config):
        """
        Function to be used for running multiple vqe runs in parallel.
        """
        atom, hamiltonian, ansatz = self._set_atom(atom_config)

        # Setup VQE
        _vqe = VQE(ansatz=ansatz, optimizer=self.optimizer, estimator=self.estimator)

        # Initialize parameters
        if not self.init_parameters:
            init_parameters = self._init_parameters()
        else:
            init_parameters = self.init_parameters
        _vqe.initial_point = init_parameters

        # Compute result
        vqe_result = _vqe.compute_minimum_eigenvalue(hamiltonian)
        parameters = list(vqe_result.optimal_parameters.values())

        return vqe_result.optimal_value, parameters

    def run_parallel(self,
                atom_configs: list[str],
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

            # Compute configs in parallel

            self.optimizer = optimizer
            self.estimator = estimator
            self.init_parameters = init_parameters

            with Pool() as pool:
                results = pool.map(self._single_vqe, atom_configs)

            # Aggregate into lists
            energies = []
            parameters_list = []

            for energy, params in results:
                energies.append(energy)
                parameters_list.append(params)

            return {
                "energy": energies,
                "parameters": parameters_list
            }

    def _update_bounds(self, old, current):
        """
        Updates the optimizer bounds through a gradient progression with divergence
        x[n] = x[n-1]+δ +/- (α|δ|+β)
        Here x[n-1]+δ is continuing the current gradient, with +/- (α|δ|+β) allowing additional bound.
        """
        deltas = [b-a for a,b in zip(old, current)]
        self.ansatz.parameter_bounds = [(x+delta-(self.alpha*abs(delta)+self.beta), x+delta+(self.alpha*abs(delta)+self.beta)) for x, delta in zip(current, deltas)]

    def run_constrained(self, atom_configs, estimator: BaseEstimatorV2, alpha=1, beta=0.2,
                backend_options: dict = None, init_parameters=None, optimizer=COBYLA()):
        
        self.optimizer = optimizer
        self.estimator = estimator
        self.init_parameters = init_parameters
        self.alpha = alpha
        self.beta = beta

        energies = []
        parameters = []

        for i in range(2):
            energy, param = self._single_vqe(atom_configs[i])
            energies.append(energy)
            parameters.append(param)
        self._update_bounds(parameters[0], parameters[1])
        self.init_parameters = parameters[1]

        for atom in atom_configs[2:]:
            energy, param = self._single_vqe(atom)
            energies.append(energy)
            parameters.append(param)

            self._update_bounds(self.init_parameters, param)
            self.init_parameters = param

        return {
            "energy": energies,
            "parameters": parameters
        }

    def assign_parameters(self, atom_config, parameters):
        self._set_atom(atom_config)

        if isinstance(self.ansatz, UCCSD):
            qc = QuantumCircuit(self.ansatz.num_qubits)
            qc.append(self.ansatz.assign_parameters(parameters), range(0, self.ansatz.num_qubits))
            return qc
        else:
            return self.ansatz.assign_parameters(parameters)

    def run_direct(self, atom_config:list[str]|str, parameters:list[list[float]]|list[float]):
        """
        evaluate ansatz directly with estimator and given parameters.
        Todo: support hardware backends
        """
        if type(atom_config)==list:
            with Pool() as pool:
                results = pool.map(run_estimator, [[a, p, self.ansatz, self.mapper] for a, p in zip(atom_config, parameters)])
                results = [float(r) for r in results]
        else:
            results = self._run_estimator((atom_config, parameters))
        return results
