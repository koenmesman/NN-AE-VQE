import sys
import os
from qiskit.circuit.library import efficient_su2
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator, StatevectorEstimator
import matplotlib.pyplot as plt
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from QAE import QAE
from VQEBase import VQEExtended
from Utils import load


# get distance from config
def to_distance(config:str):
    "find 6th number in config and return as float"
    dist = re.findall("[0-9]|\.", config)
    return float("".join(dist[5:]))

# Data
qae_file = "../data/QAE_HH.json"
ref_file = "../data/vqe_data_HH.json"
base = 4
target = 3
compression = "{}_{}".format(base, target)
qae_data = load(qae_file)[compression]

# Find best QAE setup
acc = [d["accuracy"] for d in qae_data]
qae_parameters = qae_data[acc.index(min(acc))]["parameters"]
print("QAE error :", min(acc))

# Load reference data
validation_data = load(ref_file)
configs = validation_data['exact']['points']
points = [to_distance(c) for c in configs]

exact_energies = validation_data['exact']['energy']
vqe_energies = validation_data['VQE-UCCSD']['energy']
vqe_parameters = validation_data['VQE-UCCSD']['parameters']

# Set encoder circuit
ansatz = efficient_su2(4, reps=1)
encoder = ansatz.assign_parameters(qae_parameters)

# Generate energies with QAE encoding
E_QAE = []
E_repulsion = []
estimator = Estimator()

for p, vqe_param in zip(configs, vqe_parameters):
    vqe = VQEExtended()
    vqe_state = vqe.assign_parameters(p, vqe_param)

    # Encode and decode reference VQE state
    qc = QuantumCircuit(base)
    qc.append(vqe_state, range(base))
    qc.append(encoder, range(base))
    qc.reset(range(base)[target:])
    qc.append(encoder.inverse(), range(base))
    
    # Run estimation
    H = vqe.hamiltonian
    est = estimator.run([qc], [H]).result()

    E_repulsion.append(vqe.atom.problem.nuclear_repulsion_energy)
    E_QAE.append(est.values[0])

# Add repulsion energies
E_QAE = [e+r for e, r in zip(E_QAE, E_repulsion)]
vqe_energies = [e+r for e, r in zip(vqe_energies, E_repulsion)]

error = [abs(enc-ref) for enc, ref in zip(E_QAE, vqe_energies)]
chem_acc = [0.0015]*len(points)


# Plot obtained energies
plt.plot(points, E_QAE, 'x', label="encoded")
plt.plot(points, vqe_energies, '.', label="vqe reference")
plt.ylabel("Energy (Hartree)")
plt.xlabel("Atomic distance (Angstrom)")
plt.legend()
plt.savefig("../data/QAE_validation_HH_{}_{}.png".format(base, target))
plt.show()

# Plot errors
plt.plot(points, chem_acc, label="chemical accuracy")
plt.plot(points, error, '.', label="encoding error")
plt.ylabel("Energy (Hartree)")
plt.xlabel("Atomic distance (Angstrom)")
plt.yscale("log")
plt.legend()
plt.savefig("../data/QAE_validation_HH_{}_{}_abs_error.png".format(base, target))