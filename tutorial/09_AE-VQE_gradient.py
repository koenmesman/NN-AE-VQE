from qiskit.primitives import Estimator, StatevectorEstimator
from qiskit.circuit.library import efficient_su2
from qiskit import QuantumCircuit
import numpy as np

import __init__
from Utils import store_vqe, load
from VQEBase import VQEExtended


# Data
ref_file = "../data/vqe_data_HH.json"
vqe_file = "../data/aevqe_data_HH.json"
qae_file = "../data/QAE_HH.json"
base = 4
target = 3
compression = "{}_{}".format(base, target)
qae_data = load(qae_file)[compression]
ref_data = load(ref_file)["VQE-UCCSD"]


# Find best QAE setup
acc = [d["accuracy"] for d in qae_data]
qae_parameters = qae_data[acc.index(min(acc))]["parameters"]
print("QAE error :", min(acc))

# Define ae-vqe ansatz
reps = 3
encoder = efficient_su2(base, reps=1).assign_parameters(qae_parameters)
VQE_ansatz = efficient_su2(target, reps=reps)

ansatz = QuantumCircuit(base)
ansatz.append(VQE_ansatz, range(target))
ansatz.append(encoder.inverse(), range(base))

vqe = VQEExtended(ansatz=ansatz)
estimator = StatevectorEstimator()

num_points = 100

# Get best previous parameters as new initial parameters
old_results = load(vqe_file)[compression][f"EfficientSU2-{reps}"][0]
errors = [abs(ae - ref) for ae, ref in zip(old_results["energy"], ref_data["energy"])]
init_parameters = old_results["parameters"][errors.index(min(errors))]

"""
Make sure β>0 allowing divergence in case the gradient is 0. Set α>0 for dynamic bound update.
Might be wise to design α depending on the number of points, e.g., 5/num_points.
Low divergence allows for closer matched parameters, but can constrain solution quality.
"""
alpha=10/num_points
beta=0.2

# Run computations with dynamic optimizer bound update, Should take about 5-6 minutes.
atoms = [f"H 0 0 0; H 0 0 {i}" for i in np.linspace(0.2, 3, num_points)]
result = vqe.run_constrained(atoms, alpha=alpha, beta=beta, estimator=estimator, init_parameters=None)

data = {compression:{f"EfficientSU2-{reps}-grad":{"points":atoms, "energy":result['energy'], 'parameters':result['parameters']}}}

store_vqe(vqe_file, data)