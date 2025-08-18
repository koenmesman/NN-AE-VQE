from qiskit.primitives import Estimator, StatevectorEstimator
from qiskit.circuit.library import efficient_su2, TwoLocal
from qiskit import QuantumCircuit
import numpy as np

import __init__
from Utils import store_vqe, load, store_aevqe
from VQEBase import VQEExtended
from Gates import U3TwoQubit, UniversalTwoQubit, ZZ
from qiskit_algorithms.optimizers import L_BFGS_B

# Data
vqe_file = "../data/aevqe_data_HH_big.json"
qae_file = "../data/QAE_HH.json"
base = 4
target = 3
compression = "{}_{}".format(base, target)
qae_data = load(qae_file)[compression]

# Find best QAE setup
acc = [d["accuracy"] for d in qae_data]
qae_parameters = qae_data[acc.index(min(acc))]["parameters"]
print("QAE error :", min(acc))

#__________________________________________________#

"""

# Define ae-vqe ansatz
reps = 2
encoder = efficient_su2(base, reps=1).assign_parameters(qae_parameters)
VQE_ansatz = efficient_su2(target, reps=reps)

ansatz = QuantumCircuit(base)
ansatz.append(VQE_ansatz, range(target))
ansatz.append(encoder.inverse(), range(base))

vqe = VQEExtended(ansatz=ansatz)
estimator = StatevectorEstimator()

num_points = 100

# Should take about 1-2 minutes.
atoms = [f"H 0 0 0; H 0 0 {i}" for i in np.linspace(0.2, 3, num_points)]
result = vqe.run_parallel(atoms, estimator=estimator)

data = {compression:{f"{ansatz.name}-{reps}":{"points":atoms, "energy":result['energy'], 'parameters':result['parameters']}}}

store_vqe(vqe_file, data)
"""
#__________________________________________________#

# Define a different ae-vqe ansatz
reps = 2
encoder = efficient_su2(base, reps=1).assign_parameters(qae_parameters)

VQE_ansatz = TwoLocal(num_qubits=target, rotation_blocks=["rx", "ry"],
entanglement_blocks="cx", entanglement='circular', reps=reps, name="rxry_cx_circ")

ansatz = QuantumCircuit(base)
ansatz.append(VQE_ansatz, range(target))
ansatz.append(encoder.inverse(), range(base))

vqe = VQEExtended(ansatz=ansatz)
estimator = StatevectorEstimator()

num_points = 20

# Should take about 1-2 minutes.
atoms = [f"H 0 0 0; H 0 0 {i}" for i in np.linspace(0.2, 3, num_points)]
result = vqe.run_parallel(atoms, estimator=estimator, optimizer=L_BFGS_B())

data = {compression:{"{}-{}".format(VQE_ansatz.name, reps):{"points":atoms, "energy":result['energy'],
 'parameters':result['parameters'], "evaluations":result["evaluations"]}}}

store_aevqe(vqe_file, data)