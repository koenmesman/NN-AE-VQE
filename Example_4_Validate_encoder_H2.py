#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:54:30 2024

@author: kmesman
"""

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
import sys
sys.path.append("../src")

from ChemInstance import Instance
from QAE import Encoder
from Utils import load
from Ansatz import TwoLocalU3
from qiskit.primitives import Estimator, StatevectorEstimator


ELEMENTS=["H", "H"]
SAVE_FILE = "../data/Energies_{}_{}.json".format(*ELEMENTS)
estimator = Estimator()



references = load(SAVE_FILE)
test_points = references["UCCSD_points"]
vqe_parameters = references["UCCSD_parameters"]
vqe_energies = references["UCCSD"]

enc = Encoder(file="../data/qae_config_{}_{}.json".format(4, 3), ansatz=TwoLocalU3)
inst = Instance(ELEMENTS, method="UCCSD")
inst.est = StatevectorEstimator


enc_qc = enc.bound_encoder
dec_qc = enc_qc.inverse()
qubits = inst.num_qubits

E_QAE = []
E_vqe = []

for point, vqe_param, vqe_e in zip(test_points, vqe_parameters, vqe_energies):
    inst.run(point, vqe_param)
    vqe_state = inst.bound_vqe

    qc = QuantumCircuit(qubits)
    qc.append(vqe_state, range(qubits)) # VQE
    qc.append(enc_qc, range(qubits))
    qc.reset(3)
    qc.append(dec_qc, range(qubits))
    ansatz=qc
    
    H = inst.hamiltonian
    E_repulsion = inst.problem.nuclear_repulsion_energy
    est = estimator.run([ansatz], [H]).result()
    vqe_e = estimator.run([vqe_state], [H]).result().values[0]

    E_QAE.append(est.values[0] + E_repulsion)
    E_vqe.append(vqe_e + E_repulsion)
    
    
plt.plot(test_points, E_QAE, 'x')
plt.plot(test_points, E_vqe, '.')
plt.show()