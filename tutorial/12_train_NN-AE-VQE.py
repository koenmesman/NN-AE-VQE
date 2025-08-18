import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import efficient_su2, TwoLocal
from qiskit import QuantumCircuit
import tensorflow as tf
import os

import __init__
from NeuralNetwork import NeuralNetwork
from Utils import load, to_distance
from VQEBase import VQEExtended
from Gates import U3TwoQubit, UniversalTwoQubit

# Data
exect_file = "../data/exect_data.json"
aevqe_file = "../data/aevqe_data_HH_tests.json"
base = 4
target = 3
reps=1

compression = "{}_{}".format(base, target)
ansatz = "TwoLocalUniversalU3_circ-{}-grad".format(reps)

exact_data = load(exect_file)["exact"]
exact_e = exact_data["energy"][0]
aevqe_data = load(aevqe_file)[compression][ansatz]
ref_energies = aevqe_data["energy"]
aevqe_par = aevqe_data["parameters"]
configs = aevqe_data['points']

points = [to_distance(c) for c in configs]

# Get neural network predictions
hparams = {
        "NUM_UNITS": 64,
        "NUM_LAYERS": 6,
        "DROPOUT": 0.05,
        "VALIDATE_SPLIT": 0.3,
        "EPOCHS": 2000,
        "LEARNING_RATE": 1e-3,
        "BATCH_SIZE": 32,
        "PATIENCE": 500,  # Recommended: ≈ EPOCHS/4 for small datasets, EPOCHS/10 for large datasets
    }

checkpoint_path = "../data/training_01.ckpt"

nn = NeuralNetwork()

# first train
nn.run(x_in=points, y_in=aevqe_par, hparams = hparams)
nn.model.save_weights(checkpoint_path.format(epoch=0))

# load instead
"""
if os.path.isdir(checkpoint_path):
    checkpoint_dir = checkpoint_path
else:
    checkpoint_dir = os.path.dirname(checkpoint_path)

latest = tf.train.latest_checkpoint(checkpoint_dir)
if latest is None:
    raise FileNotFoundError(f"No checkpoints found in: {checkpoint_dir}")

nn.output_shape = len(aevqe_data["parameters"][0])
nn.hparams = hparams
nn._build_model()

status = nn.model.load_weights(latest)
"""

tensors = nn.model.predict(points[0:1000:100])
predicted_par = [nn.invert_normalize(params) for params in tensors]

# Find best QAE setup
qae_file = "../data/QAE_HH.json"
qae_data = load(qae_file)[compression]

acc = [d["accuracy"] for d in qae_data]
qae_parameters = qae_data[acc.index(min(acc))]["parameters"]

encoder = efficient_su2(base, reps=1).assign_parameters(qae_parameters)

VQE_ansatz = TwoLocal(num_qubits=target, rotation_blocks="u",
entanglement_blocks=U3TwoQubit(), entanglement='circular', reps=reps, name="TwoLocalU3")

ansatz = QuantumCircuit(base)
ansatz.append(VQE_ansatz, range(target))
ansatz.append(encoder.inverse(), range(base))

vqe = VQEExtended(ansatz=ansatz)


num_par = 8
aevqe_par_transpose = np.array(aevqe_par[0:1000:100]).transpose()
predict_transpose = np.array(predicted_par).transpose()

for par, pred_par in zip(aevqe_par_transpose[:num_par], predict_transpose[:num_par]):
    plt.plot(points[0:1000:100], par, 'x', label="training")
    plt.plot(points[0:1000:100], pred_par, label="prediction")
plt.savefig("../data/param_prediction_HH.png")
plt.show()


if __name__ == "__main__":
    results = vqe.run_direct(configs[0:1000:100], predicted_par)

error = [abs(res-ref) for res, ref in zip(results, exact_e[0:1000:100])]
print(error)
chem_acc = [0.0015]*len(points[0:1000:100])

# Plot errors
plt.plot(points[0:1000:100], chem_acc, label="chemical accuracy")
plt.plot(points[0:1000:100], error, '.', label="neural network error")
plt.ylabel("Energy (Hartree)")
plt.xlabel("Atomic distance (Angstrom)")
plt.yscale("log")
plt.legend()
plt.savefig("../data/NNAEQAE_validation_HH_{}_{}_abs_error.png".format(base, target))

plt.show()