import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from VQEBase import VQEExtended
from minimal_VQE_class import Instance

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import (
    JordanWignerMapper,
    BravyiKitaevSuperFastMapper,
    ParityMapper,
)
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
#from qiskit_nature.converters.second_quantization import QubitConverter

atom = "H 0 0 0; H 0 0 1"

vqe = VQEExtended()
result = vqe.run_exact(atom)
print(result)

result = vqe.run(atom)
print(result)