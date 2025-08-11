import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import efficient_su2, TwoLocal
from qiskit import QuantumCircuit

import __init__
from NeuralNetwork import NeuralNetwork
from Utils import load, to_distance
from VQEBase import VQEExtended
from Gates import U3TwoQubit, UniversalTwoQubit

# Data
aevqe_file = "../data/aevqe_data_HH_tests.json"
base = 4
target = 3
reps=1

compression = "{}_{}".format(base, target)
ansatz = "TwoLocalU3-{}-init".format(reps)

aevqe_data = load(aevqe_file)[compression][ansatz]
ref_energies = aevqe_data["energy"]
aevqe_par = aevqe_data["parameters"]
configs = aevqe_data['points']

points = [to_distance(c) for c in configs]

# Get neural network predictions
hparams = {
        "NUM_UNITS": 32,
        "NUM_LAYERS": 3,
        "DROPOUT": 0.3,
        "VALIDATE_SPLIT": 0.2,
        "EPOCHS": 300,
        "LEARNING_RATE": 1e-3,
        "BATCH_SIZE": 8,
        "PATIENCE": 75,  # Recommended: ≈ EPOCHS/4 for small datasets, EPOCHS/10 for large datasets
    }

nn = NeuralNetwork()
nn.run(x_in=points, y_in=aevqe_par, hparams = hparams)

tensors = nn.model.predict(points)
predicted_par = [nn.invert_normalize(params) for params in tensors]


# Find best QAE setup
qae_file = "../data/QAE_HH.json"
qae_data = load(qae_file)[compression]

acc = [d["accuracy"] for d in qae_data]
qae_parameters = qae_data[acc.index(min(acc))]["parameters"]
print("QAE error :", min(acc))

encoder = efficient_su2(base, reps=1).assign_parameters(qae_parameters)

VQE_ansatz = TwoLocal(num_qubits=target, rotation_blocks="u",
entanglement_blocks=U3TwoQubit(), entanglement='circular', reps=reps, name="TwoLocalU3")

ansatz = QuantumCircuit(base)
ansatz.append(VQE_ansatz, range(target))
ansatz.append(encoder.inverse(), range(base))

vqe = VQEExtended(ansatz=ansatz)


if __name__ == "__main__":
    results = vqe.run_direct(configs, predicted_par)


error = [abs(res-ref) for res, ref in zip(results, ref_energies)]
chem_acc = [0.0015]*len(points)

# Plot errors
plt.plot(points, chem_acc, label="chemical accuracy")
plt.plot(points, error, '.', label="neural network error")
plt.ylabel("Energy (Hartree)")
plt.xlabel("Atomic distance (Angstrom)")
plt.yscale("log")
plt.legend()
plt.savefig("../data/NNAEQAE_validation_HH_{}_{}_abs_error.png".format(base, target))

plt.show()