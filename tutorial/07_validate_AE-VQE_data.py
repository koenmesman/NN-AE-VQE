import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from Utils import load, to_distance
import matplotlib.pyplot as plt

# Data
aevqe_file = "../data/aevqe_data_HH.json"
vqe_file = "../data/vqe_data_HH.json"
base = 4
target = 3
reps=2

compression = "{}_{}".format(base, target)
ansatz = "TwoLocalU3-{}".format(reps)

vqe_data = load(vqe_file)["exact"]
aevqe_data = load(aevqe_file)
aevqe_data = aevqe_data[compression][ansatz][0]
vqe_e = vqe_data["energy"]
aevqe_e = aevqe_data["energy"]
configs = aevqe_data['points']

points = [to_distance(c) for c in configs]

error = [abs(enc-ref) for enc, ref in zip(aevqe_e, vqe_e)]
chem_acc = [0.0015]*len(points)

# Plot errors
plt.plot(points, chem_acc, label="chemical accuracy")
plt.plot(points, error, '.', label="encoding error")
plt.ylabel("Energy (Hartree)")
plt.xlabel("Atomic distance (Angstrom)")
plt.yscale("log")
plt.legend()
plt.savefig("../data/AEQAE_validation_HH_{}_{}_abs_error_grad_no_init.png".format(base, target))

plt.show()
