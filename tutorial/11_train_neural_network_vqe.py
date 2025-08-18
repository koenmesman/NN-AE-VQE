import numpy as np
import matplotlib.pyplot as plt

import __init__
from NeuralNetwork import NeuralNetwork
from Utils import load, to_distance
"""
# Data
vqe_file = "../data/vqe_data_HH_tests.json"

vqe_data = load(vqe_file)['VQE-UCCSD']
vqe_par = vqe_data["parameters"]
configs = vqe_data['points']

points = [to_distance(c) for c in configs]

vqe_par_transpose = np.array(vqe_par).transpose()

for par in vqe_par_transpose:
    plt.plot(points, par)
plt.show()

#nn = NeuralNetwork()
"""