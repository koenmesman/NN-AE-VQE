from qiskit.primitives import Estimator, StatevectorEstimator
from qiskit.circuit.library import efficient_su2
from qiskit import QuantumCircuit
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from VQEBase import VQEExtended
from Utils import load
 

atom = "H 0 0 0; H 0 0 1"

# Data
qae_file = "../data/QAE_HH.json"
base = 4
target = 3
compression = "{}_{}".format(base, target)
qae_data = load(qae_file)[compression]

# Find best QAE setup
acc = [d["accuracy"] for d in qae_data]
qae_parameters = qae_data[acc.index(min(acc))]["parameters"]
print("QAE error :", min(acc))

encoder = efficient_su2(base, reps=1).assign_parameters(qae_parameters)
VQE_ansatz = efficient_su2(target, reps=2)

ansatz = QuantumCircuit(base)
ansatz.append(VQE_ansatz, range(target))
ansatz.append(encoder.inverse(), range(base))

vqe = VQEExtended(ansatz=ansatz)
result = vqe.run_exact(atom)
print("the exact diagonalization result is:", result)

estimator = StatevectorEstimator()
result = vqe.run(atom, estimator=estimator)
print("the statevector estimator result is:", result)