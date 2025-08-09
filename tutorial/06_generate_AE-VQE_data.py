from qiskit.primitives import Estimator, StatevectorEstimator
from qiskit.circuit.library import efficient_su2
from qiskit import QuantumCircuit
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from Utils import store_vqe, load
from VQEBase import VQEExtended


# Data
vqe_file = "../data/aevqe_data_HH.json"
qae_file = "../data/QAE_HH.json"
base = 4
target = 3
compression = "{}_{}".format(base, target)
qae_data = load(qae_file)[compression]

# Find best QAE setup
acc = [d["accuracy"] for d in qae_data]
qae_parameters = qae_data[acc.index(min(acc))]["parameters"]
print("QAE error :", min(acc))

# Define ae-vqe ansatz
reps = 2
encoder = efficient_su2(base, reps=1).assign_parameters(qae_parameters)
VQE_ansatz = efficient_su2(target, reps=reps)

ansatz = QuantumCircuit(base)
ansatz.append(VQE_ansatz, range(target))
ansatz.append(encoder.inverse(), range(base))

vqe = VQEExtended(ansatz=ansatz)

num_points = 100

for i in np.linspace(0.2, 3, num_points):
    print("point: {}/3".format(i))
    atom = "H 0 0 0; H 0 0 {}".format(i)

    
    estimator = StatevectorEstimator()
    result = vqe.run(atom, estimator=estimator)
    print("Result:", result['energy'])

    data = {"EfficientSU2-{}".format(reps):{"points":atom, "energy":result['energy'], 'parameters':result['parameters']}}
    store_vqe(vqe_file, data)
    

