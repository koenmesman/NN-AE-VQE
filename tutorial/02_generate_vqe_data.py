import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from VQEBase import VQEExtended
from minimal_VQE_class import Instance
from Utils import store_vqe


from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import (
    JordanWignerMapper,
    BravyiKitaevSuperFastMapper,
    ParityMapper,
)
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

from qiskit.primitives import Estimator, StatevectorEstimator

vqe_file = "../data/vqe_data_HH.json"

num_points = 100

for i in np.linspace(0.2, 3, num_points):
    print("point: {}/3".format(i)
    atom = "H 0 0 0; H 0 0 {}".format(i)

    vqe = VQEExtended()
    result = vqe.run_exact(atom)
    print("Exact diagonalization result:", result)
    data = {"exact":{"points":atom, "energy":result}}
    store_vqe(vqe_file, data)
    
    estimator = StatevectorEstimator()
    result = vqe.run(atom, estimator=estimator)
    print("Statevector estimator result:", result['energy'])

    data = {"VQE-UCCSD":{"points":atom, "energy":result['energy'], 'parameters':result['parameters']}}
    store_vqe(vqe_file, data)
    
    

