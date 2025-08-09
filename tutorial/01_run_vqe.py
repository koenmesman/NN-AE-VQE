import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from VQEBase import VQEExtended
from qiskit.primitives import Estimator, StatevectorEstimator

atom = "H 0 0 0; H 0 0 1"

vqe = VQEExtended()
result = vqe.run_exact(atom)
print("the exact diagonalization result is:", result)

estimator = StatevectorEstimator()
result = vqe.run(atom, estimator=estimator)
print("the statevector estimator result is:", result)