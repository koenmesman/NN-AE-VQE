import matplotlib.pyplot as plt
import tensorflow as tf
import os

import __init__
from NeuralNetwork import NeuralNetwork
from Utils import load, to_distance
from VQEBase import VQEExtended

from qiskit.circuit.library import efficient_su2, TwoLocal
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator, StatevectorEstimator

base = 4
target =3
reps = 2
# original vqe data
vqe_data = load("../data/vqe_data_HH_20_tests.json")
vqe_runs = vqe_data["VQE-UCCSD"]["evaluations"]
vqe_runs = vqe_data["VQE-UCCSD"]["energy"]

configs = vqe_data["exact"]["points"]
ref_e = vqe_data["exact"]["energy"]
points = [to_distance(c) for c in configs]

# ae-vqe data
aevqe_data = load("../data/aevqe_data_HH_big.json")["4_3"]["rxry_cx_circ-2"]
aevqe_runs = aevqe_data["evaluations"]
aevqe_e = aevqe_data["energy"]

init_data = load("../data/aevqe_data_HH_big.json")["4_3"]["rxry_cx_circ-2-init"]
init_aevqe_runs = init_data["evaluations"]
init_e = init_data["energy"]

# nn-ae-vqe data
# Load NN model
checkpoint_path = "../data/training_01.ckpt"
hparams = {
        "NUM_UNITS": 32,
        "NUM_LAYERS": 4,
        "DROPOUT": 0.05,
        "VALIDATE_SPLIT": 0.3,
        "EPOCHS": 400,
        "LEARNING_RATE": 1e-3,
        "BATCH_SIZE": 32,
        "PATIENCE": 50,  # Recommended: ≈ EPOCHS/4 for small datasets, EPOCHS/10 for large datasets
    }
nn = NeuralNetwork()

if os.path.isdir(checkpoint_path):
    checkpoint_dir = checkpoint_path
else:
    checkpoint_dir = os.path.dirname(checkpoint_path)

latest = tf.train.latest_checkpoint(checkpoint_dir)
if latest is None:
    raise FileNotFoundError(f"No checkpoints found in: {checkpoint_dir}")

nn.output_shape = 18
nn.hparams = hparams
nn._build_model()

status = nn.model.load_weights(latest)

tensors = nn.model.predict(points)
predicted_par = [nn.invert_normalize(params) for params in tensors]

# Find best QAE setup
qae_file = "../data/QAE_HH.json"
qae_data = load(qae_file)["4_3"]

acc = [d["accuracy"] for d in qae_data]
qae_parameters = qae_data[acc.index(min(acc))]["parameters"]

encoder = efficient_su2(base, reps=1).assign_parameters(qae_parameters)

VQE_ansatz = TwoLocal(num_qubits=target, rotation_blocks=["rx", "ry"],
entanglement_blocks="cx", entanglement='circular', reps=reps, name="rxry_cx_circ")

ansatz = QuantumCircuit(base)
ansatz.append(VQE_ansatz, range(target))
ansatz.append(encoder.inverse(), range(base))
vqe = VQEExtended(ansatz=ansatz)

estimator = StatevectorEstimator()

# use NN as hotstart
energies = []
nn_evals = []
for par, config in zip(predicted_par, configs):
    result = vqe.run(config, estimator=estimator, init_parameters=par)
    print("Statevector estimator result:", result['energy'])

    energies.append(result['energy'])
    nn_evals.append(result["evaluations"])

fig = plt.figure(figsize =(10, 7))


import numpy as np

def acc(d):
    return [abs(e-r) for e, r in zip(d, ref_e)]

data_time = [aevqe_runs, init_aevqe_runs, nn_evals]
averages = [np.mean(d) for d in data_time]
errors = [np.std(d) for d in data_time]

data_acc = [acc(d) for d in [aevqe_e, init_e, energies]]
av_acc = [np.mean(d) for d in data_acc]
err_acc = [np.std(d) for d in data_acc]

labels = ["ae-vqe", "ae-vqe + init", "nn-ae-vqe"]
x = np.arange(3)

plt.bar(labels, averages)
plt.errorbar(labels, averages, yerr=errors, fmt='o', color="c")

plt.bar(labels, av_acc, color="g", width=width1)
plt.errorbar(x+width1, av_acc, yerr=err_acc, fmt='o', color="c")

plt.show()
