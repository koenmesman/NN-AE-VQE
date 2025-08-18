import __init__

from Utils import load, to_distance, rmse, sample_data
import matplotlib.pyplot as plt

# Data
aevqe_file = "../data/aevqe_data_HH_big.json"
vqe_file = "../data/vqe_data_HH.json"
base = 4
target = 3
reps=2

compression = "{}_{}".format(base, target)
ansatz = "rxry_cx_circ-{}-grad".format(reps)

vqe_data = load(vqe_file)["exact"]
ref_configs = vqe_data["points"]
print(len(ref_configs))

aevqe_data = load(aevqe_file)[compression]
print(aevqe_data.keys())
aevqe_data = aevqe_data[ansatz]
vqe_e = vqe_data["energy"]
aevqe_e = aevqe_data["energy"]
configs = aevqe_data['points']
points = [to_distance(c) for c in configs]

ref_d, vqe_d = sample_data(10, [ref_configs, vqe_e], [configs, points, aevqe_e])
ref_configs, vqe_e = ref_d
configs, points, aevqe_e = vqe_d

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
