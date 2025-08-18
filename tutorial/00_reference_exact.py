import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from VQEBase import VQEExtended
from Utils import store_vqe

from qiskit.primitives import Estimator, StatevectorEstimator

vqe_file = "../data/exect_data.json"

num_points = 1000
atoms = []
energies = []
for i in np.linspace(0.2, 3, num_points):
    print("point: {}/3".format(i))
    atom = "H 0 0 0; H 0 0 {}".format(i)

    vqe = VQEExtended()
    result = vqe.run_exact(atom)
    atoms.append(atom)
    energies.append(result)
    print("Exact diagonalization result:", result)
data = {"exact":{"points":atoms, "energy":energies}}
store_vqe(vqe_file, data)

    
    

