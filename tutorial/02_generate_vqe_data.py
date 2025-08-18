import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from VQEBase import VQEExtended
from Utils import store_vqe

from qiskit.primitives import Estimator, StatevectorEstimator

vqe_file = "../data/vqe_data_HH_20_test.json"

num_points = 20

for i in np.linspace(0.2, 3, num_points):
    print("point: {}/3".format(i))
    atom = "H 0 0 0; H 0 0 {}".format(i)

    vqe = VQEExtended()
    result = vqe.run_exact(atom)
    print("Exact diagonalization result:", result)
    data = {"exact":{"points":[atom], "energy":[result]}}
    store_vqe(vqe_file, data)
    
    estimator = StatevectorEstimator()
    result = vqe.run(atom, estimator=estimator)
    print("Statevector estimator result:", result['energy'])

    data = {"VQE-UCCSD":{"points":[atom], "energy":[result['energy']], 'parameters':[result['parameters']]}}
    store_vqe(vqe_file, data)
