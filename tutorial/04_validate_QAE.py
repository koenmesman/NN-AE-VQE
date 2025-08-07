import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from QAE import QAE
from VQEBase import VQEExtended
from Utils import load
from qiskit.circuit.library import efficient_su2

# Data
qae_file = "../data/QAE_HH.json"
ref_file = "../data/vqe_data_HH.json"
compression = "4_2"

# Find best QAE setup
qae_data = load(qae_file)[compression]
acc = [d["accuracy"] for d in qae_data]
qae_parameters = qae_data[acc.index(min(acc))]["parameters"]

# Load reference data
validation_data = load(ref_file)
points = validation_data['exact']['points']
exact_energies = validation_data['exact']['energies']
vqe_energies = validation_data['VQE-UCCSD']['energies']
vqe_parameters = validation_data['VQE-UCCSD']['parameters']

# Set encoder circuit
ansatz = efficient_su2(4, reps=1)
encoder = ansatz.assign_parameters(qae_parameters)

E_QAE = []
E_repulsion = []

for p, vqe_param in zip(points, vqe_parameters):
    vqe = VQEExtended()
    circ = vqe.assign_parameters(sample_points[i], sample_parameters[i])

    qc = QuantumCircuit(qubits)
    qc.append(vqe_state, range(qubits)) # VQE
    qc.append(encoder, range(qubits))
    qc.reset(3)
    qc.append(encoder.inverse(), range(qubits))
    ansatz=qc
    
    H = vqe.hamiltonian
    E_repulsion = vqe.problem.nuclear_repulsion_energy
    est = estimator.run([ansatz], [H]).result()
    vqe_e = estimator.run([vqe_state], [H]).result().values[0]

    E_QAE.append(est.values[0])
    E_repulsion.append(E_repulsion)    


# get distance from config

plt.plot(test_points, E_QAE, 'x')
plt.plot(test_points, vqe_energies, '.')
plt.show()
