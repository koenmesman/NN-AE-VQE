import __init__

from Utils import load, to_distance, rmse
import matplotlib.pyplot as plt

# Data
aevqe_file = "../data/aevqe_data_HH_20.json"
vqe_file = "../data/vqe_data_HH_20.json"
base = 4
target = 3
reps=2

compression = "{}_{}".format(base, target)
ansatz = "rxry_cx_pair-{}".format(reps)

vqe_data = load(vqe_file)["exact"]
aevqe_data = load(aevqe_file)[compression]
aevqe_data = aevqe_data[ansatz]
vqe_e = vqe_data["energy"]
aevqe_e = aevqe_data["energy"]
configs = aevqe_data['points']
print(len(aevqe_data['parameters'][0]))
points = [to_distance(c) for c in configs]

error = [abs(enc-ref) for enc, ref in zip(aevqe_e, vqe_e)]
print(rmse(error))
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
