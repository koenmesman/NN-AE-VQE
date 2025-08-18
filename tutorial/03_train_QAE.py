import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from qiskit.circuit.library import efficient_su2

from QAE import QAE
from Utils import store_qae

file = "../data/QAE_HH_20.json"

ansatz = efficient_su2(4, reps=1)

qae = QAE(ansatz, 4, 3)
res = qae.train("../data/vqe_data_HH.json", num_samples=6, tol=1e-10)
print(res)

store_qae(file, res)
