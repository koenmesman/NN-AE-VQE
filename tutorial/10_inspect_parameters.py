import matplotlib.pyplot as plt
import numpy as np

import __init__
from Utils import load, to_distance

# Data
aevqe_file = "../data/aevqe_data_HH_tests.json"
base = 4
target = 3
reps=1
num_par = -1

compression = "{}_{}".format(base, target)
ansatz = "TwoLocalU3-{}-init".format(reps)

aevqe_data = load(aevqe_file)[compression][ansatz]
aevqe_par = aevqe_data["parameters"]
configs = aevqe_data['points']

points = [to_distance(c) for c in configs]

aevqe_par_transpose = np.array(aevqe_par).transpose()

for par in aevqe_par_transpose[:num_par]:
    plt.plot(points, par)
plt.show()
